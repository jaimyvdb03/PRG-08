import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton");
const startGameButton = document.getElementById("startGameButton");
const checkButton = document.getElementById("ckeckButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const image = document.querySelector("#myimage");
const simonText = document.getElementById("simonText");
const scoreDisplay = document.getElementById("score");

const drawUtils = new DrawingUtils(canvasCtx);

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let model = null;

const gestures = ["fist", "thumbsUp", "point"];
let currentSimonGesture = null;
let score = 0;

// Load TensorFlow.js model
async function loadModel() {
    model = await tf.loadLayersModel("./model/model.json");
    console.log("Model geladen");
}

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

    console.log("HandLandmarker geladen");
};

// Start webcam
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

// Webcam predict loop
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });

        let thumb = hand[4];
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`;
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Start Simon Says
function newSimonGesture() {
    const randomIndex = Math.floor(Math.random() * gestures.length);
    currentSimonGesture = gestures[randomIndex];
    simonText.textContent = `Simon says: ${currentSimonGesture}`;
}

// Classify gesture using model
async function classifyGesture() {
    if (!results?.landmarks || results.landmarks.length === 0) {
        simonText.textContent = "❗ Geen hand gedetecteerd.";
        return;
    }

    const hand = results.landmarks[0];
    const input = hand.flatMap(p => [p.x, p.y, p.z]);
    const inputTensor = tf.tensor2d([input]);

    const prediction = model.predict(inputTensor);
    const predictionArray = await prediction.array();
    const predictedIndex = predictionArray[0].indexOf(Math.max(...predictionArray[0]));

    const labels = ["fist", "thumbsUp", "point"];
    const predictedLabel = labels[predictedIndex];

    console.log("Voorspelling:", predictedLabel);

    if (predictedLabel === currentSimonGesture) {
        score++;
        scoreDisplay.textContent = score;
        simonText.textContent = "✅ Correct! Volgende komt eraan...";
        setTimeout(() => newSimonGesture(), 1000);
    } else {
        simonText.textContent = `❌ Fout! Jij deed: ${predictedLabel}, klik start game om opnieuw te beginnen.`;
    }
}

// Event listeners
enableWebcamButton.addEventListener("click", enableCam);
startGameButton.addEventListener("click", () => {
    score = 0;
    scoreDisplay.textContent = score;
    newSimonGesture();
});
checkButton.addEventListener("click", classifyGesture);

// Start alles
if (navigator.mediaDevices?.getUserMedia) {
    loadModel().then(createHandLandmarker);
}
