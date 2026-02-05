# Goal of experiment


learning a low rank adaptation module that converts gradient updates into meta learner


Experiment is successful if we can train a layer that when a new style is introduced, it changes the gradients that the old style is not forgetten.

# How meta learner is trained

We calculate two gradients in one training. Lets say we have lora weight of style A, w_A. Then a new style B is introduced. We calculate the gradient using two sets of images, one from style A and B, the other one is from only style B. (in one batch A and B's gradients are caclualted as batched gradients).

In the setup where {A,B} bathc is used, meta laerner is not applied, in the other one it is applied. We put the gradinets caclulated from A into meta learner and train it to learn output of gradients calculated from {A,B} batch.



# steps

1. Train LorA weights using the normal B-LoRa training procedure on given B-LoRa images. 

we need to train lora weights for the weights here: "repos/MoLoRAs/data/b_lora_data"



2. Use those weights to train a meta learner that takes gradients as input and produces adapted weights as output.
3. Evaluate the meta learner by introducing new styles and checking if the old styles are preserved while learning new ones.



# Build up

1. We need to test for only one anchor style first before trying multiple styles.


# TODO:

- [] Train base LoRA weights for every style in the dataset.
- [] Implement meta learner training loop using one style as anchor and others style as new styles, leaving one for testing.
- [] Evaluate the meta learner on new style and check for forgetting.