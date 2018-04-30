/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';
import {Webcam} from './webcam';


// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.
ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);

    // Draw the preview thumbnail.
    ui.drawThumb(img, label);
  });
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

// Declare the object that contains functions that use web audio to
// make sound. We don't assign it yet because we have to do that in
// response to a user interaction.
var audio;
var audio_kick = document.getElementById('audio_kick');
var audio_snare = document.getElementById('audio_snare');
var audio_hat = document.getElementById('audio_hihat');
var STEP_COUNT = 8;
var BUTTON_SIZE = 39;
var button = document.getElementById('button');
var color = "lightgray"

var onClick = function() {

    audio_kick.load(); 
    audio_snare.load();
    audio_hihat.load();

    start_running();
};

button.addEventListener('click', onClick, false);

var STEP = 0;
// Support iOS:
// https://gist.github.com/laziel/7aefabe99ee57b16081c


// Create the data for the drum machine.
var data = {
    // `step` represents the current step (or beat) of the loop.
    step: 0,

};

// Update

// Runs every {TEMPO} milliseconds.
var TEMPO = 200;
var paused = false;

function start_running() {
    setInterval(function() {
    var old_step = STEP;

    STEP = (STEP + 1) % STEP_COUNT;

    if (old_step != STEP && STEP==0) {
        audio_kick.play();
        color="lightgray";
    }
    else if (old_step != STEP && STEP==4) {
        audio_kick.play();
        audio_snare.play();
        color="deeppink"
    }
    else if (old_step != STEP && (STEP==2 || STEP==6)) {
        audio_hat.pause()
        audio_hat.currentTime = 0
        audio_hat.play();
        color="lightgray";
    }

    }, TEMPO);

    (function draw() {
    screen.clearRect(0, 0, 40, screen.canvas.height);


    // Draw the pink square that indicates the current step (beat).
    
    for(var x = 0; x < STEP_COUNT; x++) {
        drawButton(screen,
                x,
                0,
               "gray");
    };

    console.log(STEP);

    drawButton(screen, STEP, 0, color);

    requestAnimationFrame(draw);

    })();

}


var screen = document.getElementById("screen").getContext("2d");


function buttonPosition(column, row) {
    return {
        x: BUTTON_SIZE / 2 + column * BUTTON_SIZE * 1.5,
        y: BUTTON_SIZE / 2 + row * BUTTON_SIZE * 1.5
    };
};
// **drawButton()** draws a button in `color` at `column` and `row`.
function drawButton(screen, column, row, color) {
    var position = buttonPosition(column, row);
    screen.fillStyle = color;
    screen.fillRect(position.x, position.y, BUTTON_SIZE, BUTTON_SIZE);
};

function clear() {
    data.tracks.forEach(function(track) {
        track.steps = track.steps.map(function() {
            return false
        });
    });
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

// Initialize the application.
init();
