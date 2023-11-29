<script>
        document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let isPainting = false;
    let model;
    let lastX, lastY; // variables para guardar últimas posiciones
          
    // Se establecen los caractéres a predecir por el modelo
    const characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

    // Obtenemos la posición del mouse en el lienzo
    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    // Función para poder pintar en el lienzo
    function startPainting(evt) {
        isPainting = true;
        var mousePos = getMousePos(canvas, evt);
        lastX = mousePos.x;
        lastY = mousePos.y;
        draw(mousePos.x, mousePos.y, false);
    }

    // Función para detener el proceso de pintar
    function stopPainting() {
        isPainting = false;
    }

    // Función para manejar el dibujo en el lienzo
    function draw(x, y, isDown) {
        if (isDown) {
            ctx.beginPath();
            ctx.strokeStyle = '#FFFFFF'; // color de la pintura
            ctx.lineWidth = 12;
            ctx.lineJoin = 'round';
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.closePath();
            ctx.stroke();
        }
        lastX = x; lastY = y;
    }

    // Borramos todo el contenido del lienzo y se reinician los resultados
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result').innerText = "-";
        document.getElementById('confidence').innerText = "Confidence: -";
    }

    // Cargamos el modelo TensorFlow.js
    async function loadModel() {
        console.log("Loading model...");
        model = await tf.loadLayersModel('https://raw.githubusercontent.com/Raulramf/Aprendizaje-de-M-quina/main/CNN_Final/model.json/model.json');
        console.log("Model loaded.");
    }

    // Preprocesamos el lienzo para que sea una imagen en blanco y negro de 28x28
    function preprocessCanvas(image) {
        let tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([28, 28]) 
            .mean(2)
            .expandDims(2)
            .expandDims()
            .toFloat();
        return tensor.div(255.0);
    }

    // Hacemos una predicción sobre el tensor de imagen preprocesado.
    async function makePrediction() {
        let preprocessedCanvas = preprocessCanvas(canvas);
        let predictions = await model.predict(preprocessedCanvas).data();
        let results = Array.from(predictions);
        displayPredictions(results);
    }

    // Desplegamos los resultados de predicción
    function displayPredictions(predictions) {
        const maxPrediction = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxPrediction);
        // Use the characters array to display the predicted character
        document.getElementById('result').innerText = `Predicción: ${characters[maxIndex]}`;
        document.getElementById('confidence').innerText = `Confianza: ${(maxPrediction * 100).toFixed(2)}%`;
    }

    // Event listeners para el lienzo
    canvas.addEventListener('mousedown', startPainting);
    canvas.addEventListener('mouseup', stopPainting);
    canvas.addEventListener('mouseleave', stopPainting);
    canvas.addEventListener('mousemove', function(evt) {
        var mousePos = getMousePos(canvas, evt);
        draw(mousePos.x, mousePos.y, isPainting);
    });

    // Event listeners para los botones
    document.getElementById('button_predict').addEventListener('click', makePrediction);
    document.getElementById('button_clear').addEventListener('click', clearCanvas);

    // Cargamos el modelo
    loadModel();
});

    </script>
