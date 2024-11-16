using Microsoft.ML.Data;

namespace Demo.Models;

internal class InputModel
{
    [LoadColumn(0)]
    public string VendorId { get; set; } = default!;

    [LoadColumn(1)]
    public float RateCode { get; set; }

    [LoadColumn(2)]
    public float PassengerCount { get; set; }

    [LoadColumn(3)]
    public float TripTime { get; set; }

    [LoadColumn(4)]
    public float TripDistance { get; set; }

    [LoadColumn(5)]
    public string PaymentType { get; set; } = default!;

    [LoadColumn(6)]
    public float FareAmount { get; set; }
}
