const predictionForm = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewWrapper = document.getElementById("previewWrapper");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");
const statusMessage = document.getElementById("statusMessage");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultSection = document.getElementById("result");
const labelElement = document.getElementById("label");
const confidenceElement = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceTrack = document.querySelector(".confidence-track");

const allowedTypes = ["image/png", "image/jpeg"];
const maximumFileSize = 10 * 1024 * 1024;
let previewUrl = null;

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showMessage(message, type = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`.trim();
}

function clearResult() {
    resultSection.hidden = true;
    resultSection.removeAttribute("data-label");
    resultPlaceholder.hidden = false;
    confidenceBar.style.width = "0%";
    confidenceTrack.setAttribute("aria-valuenow", "0");
}

function validateImage(file) {
    if (!allowedTypes.includes(file.type)) {
        return "Please choose a PNG or JPEG image.";
    }
    if (file.size > maximumFileSize) {
        return "The selected image is larger than 10 MB.";
    }
    return "";
}

function displaySelectedImage(file) {
    const validationError = validateImage(file);
    if (validationError) {
        showMessage(validationError, "error");
        imageInput.value = "";
        predictButton.disabled = true;
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    uploadPrompt.hidden = true;
    previewWrapper.hidden = false;
    predictButton.disabled = false;
    resetButton.hidden = false;
    showMessage("");
    clearResult();
}

imageInput.addEventListener("change", () => {
    const selectedImage = imageInput.files[0];
    if (selectedImage) {
        displaySelectedImage(selectedImage);
    }
});

dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        imageInput.click();
    }
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-active");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-active");
    });
});

dropZone.addEventListener("drop", (event) => {
    const droppedFile = event.dataTransfer.files[0];
    if (!droppedFile) return;

    const transfer = new DataTransfer();
    transfer.items.add(droppedFile);
    imageInput.files = transfer.files;
    displaySelectedImage(droppedFile);
});

resetButton.addEventListener("click", () => {
    predictionForm.reset();
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
    imagePreview.removeAttribute("src");
    uploadPrompt.hidden = false;
    previewWrapper.hidden = true;
    predictButton.disabled = true;
    resetButton.hidden = true;
    showMessage("");
    clearResult();
});

predictionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const selectedImage = imageInput.files[0];
    if (!selectedImage) {
        showMessage("Please select an image before starting the analysis.", "error");
        return;
    }

    const validationError = validateImage(selectedImage);
    if (validationError) {
        showMessage(validationError, "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    predictButton.classList.add("loading");
    predictButton.disabled = true;
    resetButton.disabled = true;
    showMessage("Analyzing the image with the CNN...", "working");
    clearResult();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        const prediction = await response.json();

        if (!response.ok || prediction.error) {
            throw new Error(prediction.error || "The server could not analyze this image.");
        }

        const confidence = Math.min(
            100,
            Math.max(0, Number(prediction.confidence_percentage))
        );
        const label = String(prediction.label_name || "Unknown");

        labelElement.textContent = label;
        confidenceElement.textContent = confidence.toFixed(2);
        resultSection.dataset.label = label.toLowerCase();
        resultPlaceholder.hidden = true;
        resultSection.hidden = false;
        confidenceTrack.setAttribute("aria-valuenow", confidence.toFixed(2));

        requestAnimationFrame(() => {
            confidenceBar.style.width = `${confidence}%`;
        });

        showMessage("Analysis completed successfully.", "success");
    } catch (error) {
        showMessage(error.message || "Unable to connect to the prediction server.", "error");
    } finally {
        predictButton.classList.remove("loading");
        predictButton.disabled = false;
        resetButton.disabled = false;
    }
});

window.addEventListener("beforeunload", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
});
