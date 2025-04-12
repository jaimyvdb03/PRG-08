import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const trainButton = document.getElementById("trainButton")
const testButton = document.getElementById("testButton")

let data = []
let trainData = []
let testData = []

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

function initData(){

    fetch("data/Gesture.json")
        .then(response => response.json())
        .then(d => {
            data = d;
            trainButton.addEventListener("click", (e) => train())
        })
        .catch(error => console.log(error))
}

function train(){

    data.sort(() => (Math.random() - 0.5))

    console.log(data)
    trainData = data.slice(0, Math.floor(data.length * 0.8))
    testData = data.slice(Math.floor(data.length * 0.8) + 1)

    for(const {points, label} of trainData){
        nn.addData(points, {label: label})
        console.log(points, label)
    }

    nn.normalizeData()
    nn.train({ epochs: 35 }, () => finishedTraining())
}

testButton.addEventListener("click", (e) => test())


async function test(){
    let labels = [...new Set(testData.map(d => d.label))]; // unieke labels
    let matrix = {};

    // Initialiseer lege matrix
    for (let actual of labels) {
        matrix[actual] = {};
        for (let predicted of labels) {
            matrix[actual][predicted] = 0;
        }
    }

    for(const {points, label: actualLabel} of testData){
        const prediction = await nn.classify(points);
        const predictedLabel = prediction[0].label;

        matrix[actualLabel][predictedLabel]++;
    }

    renderConfusionMatrix(matrix, labels);
}

function renderConfusionMatrix(matrix, labels){
    const table = document.getElementById("confusionMatrix");
    table.innerHTML = ""; // reset eerst

    // Header row
    let header = "<tr><th>Actual \\ Predicted</th>";
    for(let label of labels){
        header += `<th>${label}</th>`;
    }
    header += "</tr>";
    table.innerHTML += header;

    // Rows
    for(let actual of labels){
        let row = `<tr><th>${actual}</th>`;
        for(let predicted of labels){
            row += `<td>${matrix[actual][predicted]}</td>`;
        }
        row += "</tr>";
        table.innerHTML += row;
    }
}


function finishedTraining() {
    console.log("Training voltooid!");
    nn.save("model", () => console.log("model saved"))
}

/**
 // START THE APP
 **/
if (navigator.mediaDevices?.getUserMedia) {
    initData()
}
