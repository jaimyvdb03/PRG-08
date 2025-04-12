import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "../knn/knear.js";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const checkButton = document.getElementById("ckeckButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const image = document.querySelector("#myimage");
const drawUtils = new DrawingUtils(canvasCtx);

const knn = new kNear(1);
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
const gestureExamples = []; // Voor het verzamelen van gesture data

// Hand Landmarker Setup
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    console.log("Model loaded, you can start webcam");

    fetch("./data/Gesture.json")
        .then(response => response.json())
        .then(data => train(data))
        .catch(error => console.log(error))

    enableWebcamButton.addEventListener("click", (e) => enableCam(e));
    logButton.addEventListener("click", logGestureData);

    // Add gesture button listeners
    document.getElementById("fistButton").addEventListener("click", () => saveGesture("fist"));
    document.getElementById("thumbsUpButton").addEventListener("click", () => saveGesture("thumbsUp"));
    document.getElementById("pointButton").addEventListener("click", () => saveGesture("point"));

    // Add checkButton listener
    checkButton.addEventListener("click", classifyGesture);
};

// Start Webcam
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

function train(gestures){
    for(let gesture of gestures) {
        console.log(gesture)
        knn.learn(gesture.points, gesture.label)
    }
}

// Prediction Loop
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    let hand = results.landmarks[0];
    if (hand) {
        let thumb = hand[4];
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`;
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Save Gesture Data to k-NN
function saveGesture(label) {
    if (!results?.landmarks || results.landmarks.length === 0) return;

    const hand = results.landmarks[0];
    const flatData = hand.flatMap(({ x, y, z }) => [x, y, z]);

    // Voeg toe aan knn
    knn.learn(flatData, label);

    // Voeg ook toe aan gestureExamples
    gestureExamples.push({
        points: flatData,
        label: label
    });

    console.log(`✅ Gesture "${label}" saved.`);
}


// Classify Gesture when the 'Check' button is pressed
// Classify Gesture when the 'Check' button is pressed
function classifyGesture() {
    if (!results?.landmarks || results.landmarks.length === 0) {
        console.log("No hand detected to classify.");
        return;
    }

    const hand = results.landmarks[0];

    // Flatten the hand landmarks (x, y, z) into a single array
    const handFeatures = hand.flatMap(({ x, y, z }) => [x, y, z]);

    // Classify the current hand features using k-NN
    const predictedLabel = knn.classify(handFeatures);

    console.log(`Predicted Gesture: ${predictedLabel}`);

    // Update the gesture text on the page
    const gestureTextElement = document.getElementById("gestureText");
    gestureTextElement.textContent = `Voorspeld gebaar: ${predictedLabel}`;
}


// Log Data and Classify Current Gesture
function logGestureData() {
    const textarea = document.getElementById("dataOutput");

    if (gestureExamples.length === 0) {
        textarea.value = "⚠️ Geen gesture data verzameld.";
        console.warn("⚠️ Geen gesture data verzameld.");
        return;
    }

    const jsonOutput = JSON.stringify(gestureExamples, null, 2);
    textarea.value = jsonOutput;
    console.log("📋 Gesture data:", jsonOutput);
}


// Start the App
if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker();
}
