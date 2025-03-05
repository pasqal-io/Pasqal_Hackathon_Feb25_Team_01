var patientCount = 3; // Adjust this to the actual number of patient variables
var imageCount = 6;   // Adjust this to the actual number of image features

// Generate links between patient variables and image feature vectors
var links = [];

// Generate bidirectional links between each patient variable and each image feature
for (let x = 1; x <= patientCount; x++) {
    for (let y = 1; y <= imageCount; y++) {
        // Link from patient variable to image feature
        links.push({
            "source": `Patient Variable ${x}`,
            "target": `Image Feature ${y}`,
            "type": "positive",
            "group_in": "Patient Variables",
            "group_out": "Image Features",
            "magnitude": 0.2 + (Math.random() * 4.8) // Random magnitude between 0.5 and 5.0
        });

    }
}

// Generate node descriptions
var nodeDescriptions = {};

// Add descriptions for patient variables
for (let x = 1; x <= patientCount; x++) {
    nodeDescriptions[`Patient Variable ${x}`] = `Clinical measurement or demographic information from the patient, representing an important health indicator or characteristic that may correlate with imaging features.`;
}

// Add descriptions for image features
for (let y = 1; y <= imageCount; y++) {
    nodeDescriptions[`Image Feature ${y}`] = `Quantitative measurement extracted from medical imaging data, representing specific characteristics, patterns, or biomarkers visible in the images.`;
}

// Define two groups with distinct colors
var groupColors = {
    "Patient Variables": ["#B54667", "#ffffff", "Patient Variables"],
    "Image Features": ["#90B083", "#ffffff", "Image Features"]
};

// Define the number of nodes in each group
var padding = 100;      // Padding from the top and bottom edges
var height = document.querySelector('.graph-container').clientHeight*1.85;

// Calculate vertical spacing for each group
var verticalSpacingPatient = (height - 2 * padding) / (patientCount - 1);
var verticalSpacingImage = (height - 2 * padding) / (imageCount - 1);

// Define static coordinates for nodes
var nodeCoordinates = {};

// Assign coordinates for patient variables
for (let i = 0; i < patientCount; i++) {
    nodeCoordinates[`Patient Variable ${i + 1}`] = {
        x: -500, // Position in the left quarter of the width
        y: padding + (i * verticalSpacingPatient)
    };
}

// Assign coordinates for image features
for (let i = 0; i < imageCount; i++) {
    nodeCoordinates[`Image Feature ${i + 1}`] = {
        x: 1000, // Position in the right quarter of the width
        y: padding + (i * verticalSpacingImage)
    };
}

const specialNodes = [];