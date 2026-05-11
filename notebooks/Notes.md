- A horizontal trend suggests consistent performance across epidemic sizes,
       while an upward trend would indicate higher relative errors on larger epidemics.
     - The average relative MAE_I across all samples is also computed and reported
       in the statistics summary, but this plot reveals the distribution and any size-dependence.
     - Key metric for assessing real-world applicability, as it reflects how well
       the emulator captures the critical dynamics of the epidemic regardless of scale.
     - A good emulator should ideally show a relatively flat distribution, indicating
       that its predictive accuracy is not heavily dependent on epidemic size.
     - This is crucial for decision-making, as public health responses often hinge on
       accurately predicting the peak and overall trajectory of an outbreak, regardless of its initial conditions or scale.


    Why my training loss is above the validation loss
    Validation loss being lower than training loss is caused by regularization applied during training : dropout or data augmentation, which intentionally make the optimization task harder. 