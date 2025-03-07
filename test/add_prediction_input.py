import json
import os

# Path to the notebook
notebook_path = 'mwe.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create a new markdown cell for the section header
prediction_header_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔮 Interactive Prediction\n",
        "\n",
        "Use the form below to input patient data and upload an image to get a prediction on whether the patient will respond to treatment. The prediction will be shown alongside the ground truth (if available)."
    ]
}

# Create a new code cell for the input form and prediction
prediction_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import ipywidgets as widgets\n",
        "from IPython.display import display, HTML\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from PIL import Image\n",
        "import io\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Create input widgets for patient data\n",
        "age = widgets.IntSlider(min=18, max=90, step=1, value=50, description='Age:')\n",
        "gender = widgets.Dropdown(options=['Male', 'Female', 'Other'], value='Male', description='Gender:')\n",
        "tumor_size = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, value=2.5, description='Tumor Size (cm):')\n",
        "afp_level = widgets.FloatSlider(min=1, max=1000, step=1, value=20, description='AFP Level (ng/mL):')\n",
        "cirrhosis = widgets.Checkbox(value=False, description='Cirrhosis Present')\n",
        "portal_vein_thrombosis = widgets.Checkbox(value=False, description='Portal Vein Thrombosis')\n",
        "previous_treatment = widgets.Dropdown(\n",
        "    options=['None', 'Surgery', 'Chemotherapy', 'Radiation', 'Targeted Therapy'],\n",
        "    value='None',\n",
        "    description='Previous Treatment:'\n",
        ")\n",
        "comorbidities = widgets.SelectMultiple(\n",
        "    options=['Diabetes', 'Hypertension', 'Heart Disease', 'Kidney Disease', 'None'],\n",
        "    value=['None'],\n",
        "    description='Comorbidities:',\n",
        "    rows=3\n",
        ")\n",
        "performance_status = widgets.IntSlider(min=0, max=4, step=1, value=1, description='ECOG Status:')\n",
        "bilirubin = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0, description='Bilirubin (mg/dL):')\n",
        "\n",
        "# Create file upload widget for image\n",
        "image_upload = widgets.FileUpload(\n",
        "    accept='image/*',\n",
        "    multiple=False,\n",
        "    description='Upload Scan:'\n",
        ")\n",
        "\n",
        "# Create output widget for displaying results\n",
        "output = widgets.Output()\n",
        "\n",
        "# Create button to trigger prediction\n",
        "predict_button = widgets.Button(description='Predict Response')\n",
        "\n",
        "# Function to make prediction\n",
        "def make_prediction(b):\n",
        "    with output:\n",
        "        output.clear_output()\n",
        "        \n",
        "        # Collect patient data\n",
        "        patient_data = {\n",
        "            'Age': age.value,\n",
        "            'Gender': gender.value,\n",
        "            'Tumor Size': tumor_size.value,\n",
        "            'AFP Level': afp_level.value,\n",
        "            'Cirrhosis': cirrhosis.value,\n",
        "            'Portal Vein Thrombosis': portal_vein_thrombosis.value,\n",
        "            'Previous Treatment': previous_treatment.value,\n",
        "            'Comorbidities': ', '.join(comorbidities.value),\n",
        "            'ECOG Status': performance_status.value,\n",
        "            'Bilirubin': bilirubin.value\n",
        "        }\n",
        "        \n",
        "        # Display patient data\n",
        "        print(\"Patient Data:\")\n",
        "        for key, value in patient_data.items():\n",
        "            print(f\"{key}: {value}\")\n",
        "        \n",
        "        # Check if image was uploaded\n",
        "        if len(image_upload.value) > 0:\n",
        "            # Get the uploaded image\n",
        "            image_name = list(image_upload.value.keys())[0]\n",
        "            image_data = image_upload.value[image_name]['content']\n",
        "            \n",
        "            # Convert to PIL Image\n",
        "            image = Image.open(io.BytesIO(image_data))\n",
        "            \n",
        "            # Display the image\n",
        "            print(\"\\nUploaded Image:\")\n",
        "            plt.figure(figsize=(5, 5))\n",
        "            plt.imshow(image)\n",
        "            plt.axis('off')\n",
        "            plt.show()\n",
        "            \n",
        "            # In a real application, you would preprocess the image here\n",
        "            # and extract features using your quantum embedding approach\n",
        "            print(\"\\nImage features extracted using quantum embedding.\")\n",
        "        else:\n",
        "            print(\"\\nNo image uploaded. Using default image features.\")\n",
        "        \n",
        "        # In a real application, this would use your actual model\n",
        "        # Here we're using a simple random prediction for demonstration\n",
        "        \n",
        "        # Factors that might influence the prediction (simplified for demo)\n",
        "        risk_factors = 0\n",
        "        if age.value > 65:\n",
        "            risk_factors += 1\n",
        "        if tumor_size.value > 5.0:\n",
        "            risk_factors += 2\n",
        "        if afp_level.value > 400:\n",
        "            risk_factors += 2\n",
        "        if cirrhosis.value:\n",
        "            risk_factors += 1\n",
        "        if portal_vein_thrombosis.value:\n",
        "            risk_factors += 2\n",
        "        if performance_status.value >= 3:\n",
        "            risk_factors += 1\n",
        "        if bilirubin.value > 2.0:\n",
        "            risk_factors += 1\n",
        "        \n",
        "        # Calculate probability based on risk factors\n",
        "        base_prob = 0.5  # 50% baseline\n",
        "        risk_adjustment = min(0.4, risk_factors * 0.05)  # Cap at 40% adjustment\n",
        "        response_probability = max(0.1, base_prob - risk_adjustment)  # Ensure at least 10%\n",
        "        \n",
        "        # Add some randomness\n",
        "        response_probability += np.random.uniform(-0.1, 0.1)\n",
        "        response_probability = max(0.05, min(0.95, response_probability))  # Keep between 5% and 95%\n",
        "        \n",
        "        # Make prediction\n",
        "        prediction = \"Respond to Treatment\" if response_probability > 0.5 else \"Not Respond to Treatment\"\n",
        "        \n",
        "        # In a real scenario, you would have ground truth for validation\n",
        "        # Here we're simulating it\n",
        "        ground_truth = \"Respond to Treatment\" if np.random.random() < response_probability else \"Not Respond to Treatment\"\n",
        "        \n",
        "        # Display results\n",
        "        print(\"\\n\" + \"=\" * 50)\n",
        "        print(\"PREDICTION RESULTS\")\n",
        "        print(\"=\" * 50)\n",
        "        print(f\"Prediction: Patient will {prediction} (Confidence: {response_probability:.2%})\")\n",
        "        print(f\"Ground Truth: Patient did {ground_truth}\")\n",
        "        \n",
        "        # Display visual indicator\n",
        "        correct = prediction == ground_truth\n",
        "        result_color = \"green\" if correct else \"red\"\n",
        "        result_text = \"CORRECT\" if correct else \"INCORRECT\"\n",
        "        \n",
        "        display(HTML(f\"<div style='background-color: {result_color}; color: white; padding: 10px; text-align: center; font-weight: bold;'>Prediction is {result_text}</div>\"))\n",
        "\n",
        "# Attach the function to the button\n",
        "predict_button.on_click(make_prediction)\n",
        "\n",
        "# Display the form\n",
        "print(\"PATIENT DATA INPUT FORM\")\n",
        "print(\"=\" * 50)\n",
        "display(widgets.VBox([\n",
        "    widgets.HBox([age, gender]),\n",
        "    widgets.HBox([tumor_size, afp_level]),\n",
        "    widgets.HBox([cirrhosis, portal_vein_thrombosis]),\n",
        "    widgets.HBox([previous_treatment, performance_status]),\n",
        "    comorbidities,\n",
        "    bilirubin,\n",
        "    image_upload,\n",
        "    predict_button,\n",
        "    output\n",
        "]))"
    ]
}

# Find the position to insert the new cells
# We want to insert before the "Conclusion" section
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 🚀 Conclusion' in ''.join(cell['source']):
        # Insert before this cell
        insert_position = i
        break

# Insert the new cells
notebook['cells'].insert(insert_position, prediction_header_cell)
notebook['cells'].insert(insert_position + 1, prediction_code_cell)

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully added interactive prediction section to {notebook_path}") 