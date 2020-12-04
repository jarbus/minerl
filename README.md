# MineRL Training Code

Code for a [MineRL](https://minerl.io) agent.

# Installation

`pip install gym`
`pip install minerl`

Get the dataset (60GB) and install in data/: [https://minerl.io/dataset/](https://minerl.io/dataset/)

# Usage

To run a training loop, then evaluate:
`python train.py`

# Action Space

11 discrete actions. Total list of actions:

  0. cam left
  1. cam right
  2. cam up
  3. cam down
  4. place + jump
  5. place
  6. forward + attack
  7. attack
  8. forward + jump
  9. forward
  10. jump

For Curriculum Learning, we learn a few actions at a time. We do this by keeping the same
action space, but masking actions we don't want to learn yet by multiplying by zero.

## First Task:
  * cam left
  * cam right
  * forward

## Second task:

  * cam left
  * cam right
  * forward + jump
  * forward
  * jump

## Third task:

  * cam left
  * cam right
  * cam up
  * cam down
  * attack
  * forward + jump
  * forward
  * jump


## Fourth task (full action space):


  * cam left
  * cam right
  * cam up
  * cam down
  * place + jump
  * place
  * forward + attack
  * attack
  * forward + jump
  * forward
  * jump
