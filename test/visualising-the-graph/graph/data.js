// Define patient variables
var patientVariables = [
    "Age",
    "Sex",
    "Censored_0_progressed_1",
    "Histology",
    "Stage",
    "T",
    "N",
    "M",
    "ECOG",
    "Smoking status"
]; 

// Define the prediction target
var predictionTarget = "Censored_0_progressed_1"; // 0 = censored, 1 = progressed

// Create nodes for each patient variable
var nodes = {};
patientVariables.forEach(function(variable) {
    nodes[variable] = {
        "name": variable,
        "group": "Patient Variables",
        "isPredictionTarget": variable === predictionTarget
    };
}); 