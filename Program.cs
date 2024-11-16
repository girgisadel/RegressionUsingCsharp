using Demo.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Demo;

internal static class Program
{
    static readonly string BaseDataRelativePath = @"../../../Data";
    static readonly string FullDataRelativePath = $"{BaseDataRelativePath}/taxi-fare-full.csv";
    static readonly string PreparedFullDataRelativePath = $"{BaseDataRelativePath}/prepared-taxi-fare-full.csv";
    static readonly string ModelRelativePath = $"{BaseDataRelativePath}/TaxiFareModel.zip";
    static readonly string FullDataPath = GetAbsolutePath(FullDataRelativePath);
    static readonly string PreparedFullDataPath = GetAbsolutePath(PreparedFullDataRelativePath);
    private static string ModelPath = GetAbsolutePath(ModelRelativePath);
    static readonly MLContext MLContext = new(seed: 0);

    static void Main(string[] args)
    {
        // Preprocess and save data
        PrepareData();

        // Load processed data into a data view
        var dataView = LoadData();

        // Convert to enumerable for easy manipulation
        var dataViewAsEnumerable = MLContext.Data
            .CreateEnumerable<InputModel>(dataView, reuseRowObject: false);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2, hasTopMargin: false); // Display separator

        // Report missing values in dataset
        DisplayMissingValuesReport(dataViewAsEnumerable);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2); // Display separator

        // Report distinct values for each property
        DisplayDistinctValuesReport(dataViewAsEnumerable, categoricalPropertiesOnly: false);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2); // Display separator

        // Display frequency count for PaymentType
        DisplayFrequencyCounts(dataViewAsEnumerable, nameof(InputModel.PaymentType), "Payment Type");

        // Display frequency count for PassengerCount
        DisplayFrequencyCounts(dataViewAsEnumerable, nameof(InputModel.PassengerCount), "Passenger Count");

        // Display frequency count for RateCode
        DisplayFrequencyCounts(dataViewAsEnumerable, nameof(InputModel.RateCode), "Rate Code");

        // Display frequency count for VendorId
        DisplayFrequencyCounts(dataViewAsEnumerable, nameof(InputModel.VendorId), "Vendor Id");

        // Filter out 'UNK' PaymentType
        dataViewAsEnumerable = dataViewAsEnumerable
            .Where(x => x.PaymentType != "UNK");

        // Load filtered data into data view
        var finalData = MLContext.Data.LoadFromEnumerable(dataViewAsEnumerable);

        // Shuffle the filtered data
        finalData = MLContext.Data.ShuffleRows(finalData);

        // Remove FareAmount outliers
        finalData = MLContext.Data.FilterRowsByColumn(finalData, nameof(InputModel.FareAmount), lowerBound: 1, upperBound: 150);

        // Remove rows with 0 PassengerCount
        finalData = MLContext.Data.FilterRowsByColumn(finalData, nameof(InputModel.PassengerCount), lowerBound: 1);

        // Define transformation pipeline
        var pipeline = MLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(InputModel.FareAmount))

            // Drop RateCode column
            .Append(MLContext.Transforms.DropColumns(nameof(InputModel.RateCode)))

            // One-hot encode
            .Append(MLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(InputModel.VendorId)))
            .Append(MLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(InputModel.PaymentType)))

            // Normalize
            .Append(MLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputModel.PassengerCount)))
            .Append(MLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputModel.TripTime)))
            .Append(MLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputModel.TripDistance)))

            // Concatenate features for model training
            .Append(MLContext.Transforms.Concatenate("Features", "VendorIdEncoded", "PaymentTypeEncoded",
            nameof(InputModel.PassengerCount), nameof(InputModel.TripTime), nameof(InputModel.TripDistance)));

        // Split data into training and test sets
        var trainTestSplit = MLContext.Data.TrainTestSplit(finalData, testFraction: 0.2);

        // Define regression trainer
        var trainer = MLContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features");

        // Combine pipeline and trainer
        var trainingPipeline = pipeline.Append(trainer);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2); // Display separator

        // Train model
        Console.WriteLine("Training the model...");
        var trainedModel = trainingPipeline.Fit(trainTestSplit.TrainSet);
        Console.WriteLine();

        // Make predictions on test data
        Console.WriteLine("Evaluating Model's accuracy with Test data...");
        var predictions = trainedModel.Transform(trainTestSplit.TestSet);
        Console.WriteLine();

        // Evaluate model performance
        var metrics = MLContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
        DisplayRegressionMetrics("Fast Tree", metrics);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2); // Display separator

        // Save the trained model
        MLContext.Model.Save(trainedModel, trainTestSplit.TrainSet.Schema, ModelPath);

        // Output model save location
        Console.WriteLine("The model is saved to {0}", Path.GetFullPath(ModelPath));

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2); // Display separator

        // Test model with a single prediction
        TestSinglePrediction(MLContext);

        DisplaySeparator(separatorChar: '*', count: 80, lines: 2, hasBottomMargin: false); // Display final separator
    }

    static void DisplayRegressionMetrics(string name, RegressionMetrics metrics)
    {
        Console.WriteLine("-------------------- Metrics for {name} regression model --------------------");
        Console.WriteLine($"- LossFn: {metrics.LossFunction:0.##}");
        Console.WriteLine($"- R2 Score: {metrics.RSquared:0.##}");
        Console.WriteLine($"- Absolute loss: {metrics.MeanAbsoluteError:#.##}");
        Console.WriteLine($"- Squared loss: {metrics.MeanSquaredError:#.##}");
        Console.WriteLine($"- RMS loss: {metrics.RootMeanSquaredError:#.##}");
    }

    static void PrepareData()
    {
        var processedLines = File.ReadLines(FullDataPath)
            // Process data lines, skipping header
            .Select((line, index) => index == 0 ? line : ProcessLine(line))
            .ToList();

        // Write processed data to a new file
        File.WriteAllLines(PreparedFullDataPath, processedLines);
    }

    static string ProcessLine(string line)
    {
        // Split the line into columns
        var columns = line.Split(',');

        for (int i = 0; i < columns.Length; ++i)
        {
            // Check for empty columns
            if (string.IsNullOrWhiteSpace(columns[i]))
            {
                columns[i] = (i == 1 || i == 2 || i == 3 || i == 4 || i == 6) ? "NaN" : string.Empty; // Handle missing values
            }
        }

        // Join columns back into a line
        return string.Join(",", columns);
    }

    static IDataView LoadData()
    {
        var dataView = MLContext.Data
            // Load processed data file
            .LoadFromTextFile<InputModel>(PreparedFullDataPath, hasHeader: true, separatorChar: ',');
        return dataView;
    }

    static void TestSinglePrediction(MLContext mlContext)
    {
        // Example input data for prediction
        var taxiTripSample = new InputModel()
        {
            VendorId = "VTS",
            RateCode = 1,
            PassengerCount = 1,
            TripTime = 1140,
            TripDistance = 3.75f,
            PaymentType = "CRD",
            FareAmount = 0 // FareAmount is set to 0 for prediction
        };

        // Load the trained model
        var trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

        // Create prediction engine
        var predEngine = mlContext.Model.CreatePredictionEngine<InputModel, InputModelPrediction>(trainedModel);

        // Make prediction
        var resultprediction = predEngine.Predict(taxiTripSample);

        // Display predicted fare
        Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 15.5");
    }

    static void DisplayMissingValuesReport(IEnumerable<InputModel> dataViewAsEnumerable)
    {
        Console.WriteLine("-------------------- Missing Values Report --------------------");
        var properties = typeof(InputModel).GetProperties(); // Get properties of InputModel
        foreach (var property in properties)
        {
            var missingCount = dataViewAsEnumerable
                .Count(x =>
                {
                    if (property.PropertyType == typeof(float)) // Check for missing float values
                    {
                        return float.IsNaN((float?)property.GetValue(x) ?? float.NaN);
                    }

                    if (property.PropertyType == typeof(string)) // Check for missing string values
                    {
                        return string.IsNullOrWhiteSpace((string?)property.GetValue(x));
                    }

                    return property.GetValue(x) == null; // Check for null values
                });

            Console.WriteLine($"Column: [{property.Name}] contains `{missingCount}` missing value(s).");
        }
    }

    static void DisplayDistinctValuesReport(IEnumerable<InputModel> dataViewAsEnumerable, bool categoricalPropertiesOnly = false)
    {
        Console.WriteLine($"-------------------- Distinc Values Report{(!categoricalPropertiesOnly ? " (For all properties)" : string.Empty)} --------------------");

        var properties = typeof(InputModel).GetProperties().ToList();

        if (categoricalPropertiesOnly)
        {
            properties = properties.Where(p => p.PropertyType == typeof(string)).ToList(); // Filter for categorical properties if specified
        }

        foreach (var property in properties)
        {
            var distinctValues = dataViewAsEnumerable
                .Select(x => property.GetValue(x))
                .Where(x =>
                {
                    if (property.PropertyType == typeof(float)) // Filter out NaN values
                    {
                        return !float.IsNaN((float?)x ?? float.NaN);
                    }

                    if (property.PropertyType == typeof(string)) // Filter out empty strings
                    {
                        return !string.IsNullOrWhiteSpace((string?)x);
                    }

                    return x != null;
                })
                .Distinct()
                .ToList();
            Console.WriteLine($"Column: [{property.Name}] contains `{distinctValues.Count}` distinct value(s).");
            distinctValues.Take(7).ToList().ForEach(x =>
            {
                Console.WriteLine($" - {x}");
            });
            if (distinctValues.Count > 7)
            {
                Console.WriteLine(" - ...");
            }
        }
    }

    static void DisplayFrequencyCounts(IEnumerable<InputModel> data, string propertyName, string? displayName = null)
    {
        var dataWithNoMissingValues = data.Where(x =>
        {
            return !float.IsNaN(x.TripDistance) &&
            !float.IsNaN(x.TripTime) &&
            !float.IsNaN(x.PassengerCount) &&
            !string.IsNullOrWhiteSpace(x.PaymentType) &&
            !string.IsNullOrWhiteSpace(x.VendorId) &&
            !float.IsNaN(x.RateCode) &&
            !float.IsNaN(x.FareAmount);
        });

        var frequencies = dataWithNoMissingValues.GroupBy(x => typeof(InputModel).GetProperty(propertyName)?.GetValue(x))
            .Select(x => new { Value = x.Key, Count = x.Count() })
            .ToList();

        var whole = dataWithNoMissingValues.Count();

        Console.WriteLine($"Frequency of [{displayName ?? propertyName}]'s Values:");
        foreach (var frequency in frequencies)
        {
            Console.WriteLine($"{frequency.Value}: {frequency.Count} - Percentage ({CalculatePercentage(frequency.Count, whole)})");
        }
    }

    static string GetAbsolutePath(string relativePath)
    {
        var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
        string assemblyFolderPath = _dataRoot.Directory!.FullName;

        string fullPath = Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    static string CalculatePercentage(int part, int whole)
    {
        if (whole == 0) return "0.000%";
        double percentage = ((double)part / whole) * 100.0;
        return percentage.ToString("F5") + "%";
    }

    static void DisplaySeparator(char separatorChar = '=', int count = 20, int lines = 1, bool hasTopMargin = true, bool hasBottomMargin = true)
    {
        if (hasTopMargin)
        {
            Console.WriteLine();
        }
        while (lines-- > 0)
        {
            Console.WriteLine(new string(separatorChar, count));
        }
        if (hasBottomMargin)
        {
            Console.WriteLine();
        }
    }
}
