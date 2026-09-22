const predictionForm = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");

const resultSection = document.getElementById("result");
const labelElement = document.getElementById("label");
const confidenceElement = document.getElementById("confidence");
const statusMessage = document.getElementById("statusMessage");


imageInput.addEventListener("change", function () {
    const selectedImage = imageInput.files[0];

    if (selectedImage) {
        imagePreview.src = URL.createObjectURL(selectedImage);
        imagePreview.style.display = "block";
        resultSection.style.display = "none";
    }
});


predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const selectedImage = imageInput.files[0];

    if (!selectedImage) {
        statusMessage.textContent = "Please select an image.";
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    statusMessage.textContent = "Predicting...";
    resultSection.style.display = "none";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const prediction = await response.json();

        if (prediction.error) {
            statusMessage.textContent = prediction.error;
            return;
        }

        labelElement.textContent = prediction.label_name;
        confidenceElement.textContent =
            prediction.confidence_percentage;

        statusMessage.textContent = "";
        resultSection.style.display = "block";

    } catch (error) {
        statusMessage.textContent =
            "Unable to connect to the prediction server.";
    }
});