# TabularDataGAN

This repository explores the generation of tabular data using a Generative Adversarial Network (GAN). To ensure that the generated data aligns with the distribution of the original dataset, we apply a technique inspired by Physics-Informed Neural Networks (PINN).

In this approach, statistical metrics are computed for each variable, such as the median, 25th and 75th percentiles, and other relevant values. Additionally, dataset-wide correlation measures are included, such as linear correlation, autocorrelation, and the covariance matrix between different variables.

The developed GAN models integrate these statistical metrics into the cost function, allowing the training process to adjust the model weights to better match the distributions of the original data.

This work also studies the [SDV library](https://github.com/sdv-dev/SDV) and compares its models with the ones developed in this repository.

The tests performed are also presented as a research work in the document Tabular_GAN_PINN.pdf.