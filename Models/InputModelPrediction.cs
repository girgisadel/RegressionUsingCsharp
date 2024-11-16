using Microsoft.ML.Data;

namespace Demo.Models;

internal class InputModelPrediction
{
    [ColumnName("Score")]
    public float FareAmount;
}
