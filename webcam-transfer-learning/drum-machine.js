// Declare the object that contains functions that use web audio to
// make sound. We don't assign it yet because we have to do that in
// response to a user interaction.
var audio;
var audio_kick = new Audio('kick.mp3');
var audio_snare = new Audio('snare.mp3');
var audio_hat = new Audio('hihat.mp3');
var STEP_COUNT = 8;
var BUTTON_SIZE = 39;

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

export function start_running() {
    setInterval(function() {
    var old_step = STEP;

    STEP = (STEP + 1) % STEP_COUNT;

    if (old_step != STEP && STEP==0) {
        audio_kick.play();
    }
    else if (old_step != STEP && STEP==4) {
        audio_kick.play();
        audio_snare.play();
    }
    else if (old_step != STEP && (STEP==2 || STEP==6)) {
        audio_hat.pause()
        audio_hat.currentTime = 0
        audio_hat.play();
    }

    }, TEMPO);

    (function draw() {
    screen.clearRect(0, 0, screen.canvas.width, screen.canvas.height);


    // Draw the pink square that indicates the current step (beat).
    
    for(var x = 0; x < STEP_COUNT; x++) {
        drawButton(screen,
                x,
                0,
               "lightgray");
    };

    console.log(STEP);

    drawButton(screen, STEP, 0, "deeppink");

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

