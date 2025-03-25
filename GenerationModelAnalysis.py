import os
import numpy as np
import pandas as pd
import pickle
import itertools

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
import sklearn.metrics as skm

import torch
import torch.nn as nn
import torch.optim as optim

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

from DataProcessing import AdultDataPreprocessor, MaternalDataPreprocessor

from Models import Generators, Discriminators, LossFunctions
import Models.utils as utils


class GeneratorExperiment:
    def __init__(self, data_real_dict, num_epochs=8000, batch_size=500, generator_lr=1e-4, discriminator_lr=1e-3):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.data_real_dict = data_real_dict

    def generate_model_from_data(self, model="LinearNN", loss_method="iqr-covmat-integral", verbose=False):

        # Data Min-Max scaling
        scaler = MinMaxScaler()
        data_minmax = scaler.fit_transform(self.data_real_dict["data"])

        data_column_dim = self.data_real_dict["data"].shape[1]

        if model == "LinearNN":
            generator = Generators.EncoderDecoderGenerator(in_out_dim=data_column_dim, latent_dim=16)
            optimizer_g = optim.Adam(generator.parameters(), lr=self.generator_lr)

            # Initialize discriminator
            discriminator = Discriminators.Discriminator(input_dim=data_column_dim)
            optimizer_d = optim.Adam(discriminator.parameters(), lr=self.discriminator_lr)

            # Loss function
            criterion = nn.BCELoss()

            best_loss = np.inf

            save_nn_data_list = []
            for epoch in range(self.num_epochs):

                idx = np.random.randint(0, data_minmax.shape[0], self.batch_size)
                real_data_array = data_minmax[idx]
                real_data = torch.tensor(real_data_array, dtype=torch.float32)
                
                z = torch.randn(self.batch_size, data_column_dim)
                z_minmax = (z - z.min()) / (z.max() - z.min())
                fake_data = generator(z_minmax)

                if torch.any(torch.isnan(fake_data)) or torch.any(torch.isinf(fake_data)):
                    print("Fake data contains NaNs or Infs!")
                    break

                if torch.any(torch.isnan(real_data)) or torch.any(torch.isinf(real_data)):
                    print("Real data contains NaNs or Infs!")
                    break
                
                # Entrenar Discriminador
                optimizer_d.zero_grad()
                real_loss = criterion(discriminator(real_data), torch.ones(self.batch_size, 1))
                fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(self.batch_size, 1))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Entrenar Generador
                optimizer_g.zero_grad()
                g_loss = LossFunctions.custom_loss(fake_data, real_data, method=loss_method)
                g_loss.backward()
                # g_loss.backward()
                optimizer_g.step()

                model_saved = False
                if g_loss.item() < best_loss and epoch > 1000:
                    if os.path.exists("SaveModels/generator_NN.pth"):
                        os.remove("SaveModels/generator_NN.pth")
                    model_saved = True
                    best_loss = g_loss.item()
                    torch.save(generator.state_dict(), "SaveModels/generator_NN.pth")
                    # print(f"Epoch {epoch}: New best model saved with loss {best_loss:.6f}")

                if epoch % 100 == 0:
                    save_nn_data_list.append({
                        "epoch": epoch,
                        "discriminator_loss": d_loss.item(),
                        "generator_loss": g_loss.item(),
                        "model_saved": model_saved
                    })

                    if epoch % 500 == 0 and verbose:
                        print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

                    if utils.stop_training_func(save_nn_data_list, epoch_dist=1000):
                        break

            if verbose:
                print(f"Epoch {epoch + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            generator.load_state_dict(torch.load("SaveModels/generator_NN.pth"))
            generator.eval()

        else:
            ValueError(f"{model} not implemented")

        return generator
        

# ################################
# ### Evaluate Generator Model ###
# ################################

class EvaluateGeneratorModel:
    def __init__(self, data_real_dict, generator, model_name):
        self.data_real_dict = data_real_dict
        self.metrics_skm_dict = metrics_skm_dict
        self.model_name = model_name
        self.generator = generator

        self.ml_model_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
            ('logr', LogisticRegression(max_iter=500))
        ])

        self.eval_gen_model_dict = {}

    # Study if a ML model could distinguish between generated and real data
    def ml_difference_real_generated_data(self, num_generate_th=30, batch_data=500):
        target_col = "RealTarget"
        data_column_dim = self.data_real_dict["data"].shape[1]

        scaler = MinMaxScaler()
        scaler.fit(self.data_real_dict["data"])

        if self.model_name == "LinearNN":
            help_list = []
            for i_gen in range(num_generate_th):

                z = torch.randn(batch_data, data_column_dim)
                z_minmax = (z - z.min()) / (z.max() - z.min())
                synthetic_data = self.generator(z_minmax).detach().numpy()

                # Denormalizar
                synthetic_data = scaler.inverse_transform(synthetic_data)

                # Guardar en CSV
                synthetic_df = pd.DataFrame(synthetic_data, columns=self.data_real_dict["cols_to_study"])

                custom_syn_df = synthetic_df.copy()

                # for col_name, col_type in custom_metadata_dict.items():
                #     if col_type == "integer":
                #         custom_syn_df[col_name] = custom_syn_df[col_name].round(0).astype(int)
                #     if "round" in col_type:
                #         round_th = int(col_type.split("_")[-1])
                #         custom_syn_df[col_name] = custom_syn_df[col_name].round(round_th).astype(str).astype(float)

                custom_syn_df[target_col] = 0

                idx = np.random.randint(0, self.data_real_dict["data"].shape[0], batch_data)
                df_real_sample = self.data_real_dict["data"].loc[idx]
                df_real_sample[target_col] = 1

                df_both = pd.concat([df_real_sample, custom_syn_df], ignore_index=True)

                rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=0)

                for i, (train_idx, test_idx) in enumerate(rkf.split(df_both.index)):

                    x_train = df_both.loc[train_idx, self.data_real_dict["cols_to_study"]]
                    y_train = df_both.loc[train_idx, target_col]
                    x_test = df_both.loc[test_idx, self.data_real_dict["cols_to_study"]]
                    y_test = df_both.loc[test_idx, target_col]
                    
                    self.ml_model_pipeline.fit(x_train, y_train)
                    
                    y_pred = self.ml_model_pipeline.predict(x_test)
                    
                    help_dict = {"i_gen": i_gen, "iteration": i}
                    help_dict.update(
                        {k: v(y_test, y_pred) for k, v in self.metrics_skm_dict.items()}
                    )
                    
                    help_list.append(help_dict)

            df_gen_ml_metrics = pd.DataFrame(help_list)

        elif self.model_name == "SDV":
            df_to_sdv = self.data_real_dict["data"]
            help_list = []
            for i_gen in range(num_generate_th):

                idx = np.random.randint(0, df_to_sdv.shape[0], batch_data)
                synthetic_data = self.generator.fit(data=df_to_sdv.loc[idx].reset_index(drop=True))
                synthetic_df_sdv = self.generator.sample(num_rows=batch_data)
                synthetic_df_sdv[target_col] = 0

                idx = np.random.randint(0, df_to_sdv.shape[0], batch_data)
                df_real_sample = df_to_sdv.loc[idx]
                df_real_sample[target_col] = 1

                df_both = pd.concat([df_real_sample, synthetic_df_sdv], ignore_index=True)

                rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=0)

                for i, (train_idx, test_idx) in enumerate(rkf.split(df_both.index)):

                    x_train = df_both.loc[train_idx, self.data_real_dict["cols_to_study"]]
                    y_train = df_both.loc[train_idx, target_col]
                    x_test = df_both.loc[test_idx, self.data_real_dict["cols_to_study"]]
                    y_test = df_both.loc[test_idx, target_col]
                    
                    self.ml_model_pipeline.fit(x_train, y_train)
                    
                    y_pred = self.ml_model_pipeline.predict(x_test)
                    
                    help_dict = {"i_gen": i_gen, "iteration": i}
                    help_dict.update(
                        {k: v(y_test, y_pred) for k, v in self.metrics_skm_dict.items()}
                    )
                    
                    help_list.append(help_dict)

            df_gen_ml_metrics = pd.DataFrame(help_list)

        self.eval_gen_model_dict.update({"ml_difference_real_generated_data": df_gen_ml_metrics})
            

if __name__ == "__main__":
    all_data_list = [
        AdultDataPreprocessor().get_data(),
        MaternalDataPreprocessor().get_data()
    ]

    generative_model_list = [
        "SDV", "LinearNN"
    ]

    metrics_skm_dict = {
        "accuracy": skm.accuracy_score,
        "kappa": skm.cohen_kappa_score,
        "f1_score": skm.f1_score
    }

    data_model_list = [all_data_list, generative_model_list]
    comb_data_gen_list = list(itertools.product(*data_model_list))
    for data_dict, model_name in comb_data_gen_list:

        print(data_dict["data_name"], model_name)

        if model_name != "SDV":
            gen_experiment = GeneratorExperiment(data_real_dict=data_dict)
            generator_model = gen_experiment.generate_model_from_data()
        else:
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(data=data_dict["data"])
            generator_model = GaussianCopulaSynthesizer(sdv_metadata)

        # Evaluate generator model
        eval_gen_model = EvaluateGeneratorModel(
            data_real_dict=data_dict, generator=generator_model, model_name=model_name
        )
        eval_gen_model.ml_difference_real_generated_data()

        generator_eval_results_dict = eval_gen_model.eval_gen_model_dict

        data_name = data_dict["data_name"]
        save_dict_pickle = f"Results/GenResults_{data_name}_{model_name}.pkl"
        with open(save_dict_pickle, 'wb') as f:
            pickle.dump(generator_eval_results_dict, f)
