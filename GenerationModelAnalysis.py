import gc
import os
import time
import numpy as np
import pandas as pd
import scipy as sp
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

from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

from DataProcessing import (AdultDataPreprocessor, 
                            MaternalDataPreprocessor, 
                            TitanicDataPreprocessor, 
                            StudentDropoutDataPreprocessor, 
                            WineQualityDataPreprocessor, 
                            WisconsinDataPreprocessor)

from Models import Generators, Discriminators, LossFunctions
import Models.utils as utils

from Processing.CategoricalProcessing import CategoricalToNumericalNorm as c2nn

import warnings
warnings.filterwarnings('ignore')


class GeneratorExperiment:
    def __init__(
        self,
        data_real_dict,
        model_dict,
        num_epochs=8000,
        batch_size=500,
        generator_lr=1e-4,
        discriminator_lr=1e-3,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.data_real_dict = data_real_dict
        self.data_train = self.data_real_dict["data"][self.data_real_dict["cols_to_study"]]
        self.model_dict = model_dict

    def generate_model_from_data(self, verbose=False):        
        # Data Min-Max scaling
        scaler = MinMaxScaler()
        data_min_max = scaler.fit_transform(self.data_train)

        data_col_dim = self.data_train.shape[1]

        if model_dict["model_name"] in ["LinearNN", "CNN1D"]:
            if model_dict["model_name"] == "CNN1D":
                generator = Generators.EncoderDecoderCNN1D(input_dim=data_col_dim)
            else:
                generator = Generators.EncoderDecoderGenerator(in_out_dim=data_col_dim, latent_dim=16)
            optimizer_g = optim.Adam(generator.parameters(), lr=self.generator_lr)

            # Initialize discriminator
            discriminator = Discriminators.Discriminator(input_dim=data_col_dim)
            optimizer_d = optim.Adam(discriminator.parameters(), lr=self.discriminator_lr)

            # Loss function
            criterion = nn.BCELoss()

            best_loss = np.inf

            save_nn_data_list = []
            for epoch in range(self.num_epochs):

                idx = np.random.randint(0, data_min_max.shape[0], self.batch_size)
                real_data_array = data_min_max[idx]
                real_data = torch.tensor(real_data_array, dtype=torch.float32)

                z = torch.randn(self.batch_size, data_col_dim)
                z_min_max = (z - z.min()) / (z.max() - z.min())
                
                if model_dict["model_name"] == "CNN1D":
                    z_reshape = z_min_max.reshape(self.batch_size, data_col_dim, 1)
                    fake_data_reshape = generator(z_reshape)
                    fake_data = fake_data_reshape.reshape(self.batch_size, data_col_dim)
                else:
                    fake_data = generator(z_min_max)
                
                if torch.any(torch.isnan(fake_data)) or torch.any(torch.isinf(fake_data)):
                    print("Fake data contains NaNs or Infs!")
                    break

                if torch.any(torch.isnan(real_data)) or torch.any(torch.isinf(real_data)):
                    print("Real data contains NaNs or Infs!")
                    break

                # Discriminator
                optimizer_d.zero_grad()
                real_loss = criterion(discriminator(real_data), torch.ones(self.batch_size, 1))
                fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(self.batch_size, 1))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Generator
                optimizer_g.zero_grad()
                if self.model_dict["loss_function"] == "BCELoss":
                    g_loss = criterion(discriminator(fake_data), torch.ones(self.batch_size, 1))
                elif "both-loss" in self.model_dict["loss_function"]:
                    loss_f = self.model_dict["loss_function"].split("_")[-1]
                    bce_loss = criterion(discriminator(fake_data), torch.ones(self.batch_size, 1))
                    custom_loss = LossFunctions.custom_loss(fake_data, real_data, method=loss_f)
                    g_loss = torch.mean(torch.stack([bce_loss, custom_loss]))
                else:
                    g_loss = LossFunctions.custom_loss(fake_data, real_data, method=model_dict["loss_function"])
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
                            "model_saved": model_saved,
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
            model_name = model_dict["model_name"]
            ValueError(f"{model_name} not implemented")

        return generator


# ################################
# ### Evaluate Generator Model ###
# ################################
class EvaluateGeneratorModel:
    def __init__(self, data_class, data_real_dict, generator, model_name):
        self.data_class = data_class
        self.data_real_dict = data_real_dict
        self.data_train = self.data_real_dict["data"][self.data_real_dict["cols_to_study"]]
        self.metrics_skm_dict = metrics_skm_dict
        self.model_name = model_name
        self.generator = generator

        self.ml_model_pipeline = Pipeline([
                ("std_scaler", StandardScaler()),
                ("logr", LogisticRegression(max_iter=500)),
        ])

        self.cv_n_splits = 5
        self.cv_n_repeats = 4
        
        self.eval_gen_model_dict = {}

    def train_model(self, data_to_ml, ml_input_col, data_test=None, target_col="RealTarget", join_data=None):
        
        if data_test is None:
            data_test = data_to_ml.copy()
        
        rkf = RepeatedKFold(n_splits=self.cv_n_splits, n_repeats=self.cv_n_repeats)

        help_list = []
        for i, (train_idx, test_idx) in enumerate(rkf.split(data_to_ml.index)):
            
            x_train = data_to_ml.loc[train_idx, ml_input_col]
            y_train = data_to_ml.loc[train_idx, target_col]
            if join_data is not None:
                x_train = pd.concat([x_train, join_data[ml_input_col]], ignore_index=True)
                y_train = pd.concat([y_train, join_data[target_col]], ignore_index=True)

            if len(np.unique(y_train)) < 2:
                continue
            
            x_test = data_test.loc[test_idx, ml_input_col]
            y_test = data_test.loc[test_idx, target_col]
            
            self.ml_model_pipeline.fit(x_train, y_train)

            y_pred = self.ml_model_pipeline.predict(x_test)

            # help_dict = {"iteration": i}
            help_dict = {k: v(y_test, y_pred) for k, v in self.metrics_skm_dict.items()}
            help_list.append(help_dict)

        return help_list

    # Study if a ML model could distinguish between generated and real data
    def ml_difference_real_generated_data(self, num_generate_th=30, batch_data=500):
        target_col = "RealTarget"
        data_column_dim = self.data_train.shape[1]

        if model_dict["model_name"] in ["LinearNN", "CNN1D"]:
            
            scaler = MinMaxScaler()
            scaler.fit(self.data_train)
            
            help_list = []
            for _ in range(num_generate_th):
                z = torch.randn(batch_data, data_column_dim)
                z_min_max = (z - z.min()) / (z.max() - z.min())
                
                if model_dict["model_name"] == "CNN1D":
                    z_reshape = z_min_max.reshape(batch_data, data_column_dim, 1)
                    fake_data_reshape = self.generator(z_reshape).detach().numpy()
                    synthetic_data = fake_data_reshape.reshape(batch_data, data_column_dim)
                else:
                    synthetic_data = self.generator(z_min_max).detach().numpy()
                
                # z = torch.randn(batch_data, data_column_dim)
                # z_minmax = (z - z.min()) / (z.max() - z.min())
                # synthetic_data = self.generator(z_minmax).detach().numpy()
                synthetic_data = scaler.inverse_transform(synthetic_data)
                synthetic_df = pd.DataFrame(
                    synthetic_data, columns=self.data_real_dict["cols_to_study"]
                )
                custom_syn_df = synthetic_df.copy()

                # for col_name, col_type in custom_metadata_dict.items():
                #     if col_type == "integer":
                #         custom_syn_df[col_name] = custom_syn_df[col_name].round(0).astype(int)
                #     if "round" in col_type:
                #         round_th = int(col_type.split("_")[-1])
                #         custom_syn_df[col_name] = custom_syn_df[col_name].round(round_th).astype(str).astype(float)

                custom_syn_df[target_col] = 0

                idx = np.random.randint(0, self.data_train.shape[0], batch_data)
                df_real_sample = self.data_train.loc[idx]
                df_real_sample[target_col] = 1

                df_both = pd.concat([df_real_sample, custom_syn_df], ignore_index=True)
                
                ml_score_list = self.train_model(
                    data_to_ml=df_both, 
                    target_col="RealTarget", 
                    ml_input_col=self.data_real_dict["cols_to_study"]
                )
                help_list.extend(ml_score_list)

            df_gen_ml_metrics = pd.DataFrame(help_list)

        elif self.model_name == "SDV":
            df_to_sdv = self.data_train
            help_list = []
            for _ in range(num_generate_th):

                idx = np.random.randint(0, df_to_sdv.shape[0], batch_data)
                self.generator.fit(data=df_to_sdv.loc[idx].reset_index(drop=True))
                synthetic_df_sdv = self.generator.sample(num_rows=batch_data)
                synthetic_df_sdv[target_col] = 0

                idx = np.random.randint(0, df_to_sdv.shape[0], batch_data)
                df_real_sample = df_to_sdv.loc[idx]
                df_real_sample[target_col] = 1

                df_both = pd.concat([df_real_sample, synthetic_df_sdv], ignore_index=True)

                ml_score_list = self.train_model(
                    data_to_ml=df_both, 
                    target_col="RealTarget", 
                    ml_input_col=self.data_real_dict["cols_to_study"]
                )
                help_list.extend(ml_score_list)

            df_gen_ml_metrics = pd.DataFrame(help_list)

        self.eval_gen_model_dict.update(
            {"ml_difference_real_generated_data": df_gen_ml_metrics}
        )

    def ml_train_model_comparison(self, num_generate_th=30, batch_data=1000):
        
        target_col = self.data_class.target_column
        data_column_dim = self.data_train.shape[1]
        category_inter = self.data_class.all_category_inter[target_col]
        ml_input_list = [col for col in self.data_train.columns if target_col not in col]
        
        df_to_ml = self.data_train
        
        if model_dict["model_name"] in ["LinearNN", "CNN1D"]:
            
            scaler = MinMaxScaler()
            scaler.fit(self.data_train)
            
            help_list = []
            for _ in range(num_generate_th):
                
                z = torch.randn(batch_data, data_column_dim)
                z_min_max = (z - z.min()) / (z.max() - z.min())
                
                if model_dict["model_name"] == "CNN1D":
                    z_reshape = z_min_max.reshape(batch_data, data_column_dim, 1)
                    fake_data_reshape = self.generator(z_reshape).detach().numpy()
                    synthetic_data = fake_data_reshape.reshape(batch_data, data_column_dim)
                else:
                    synthetic_data = self.generator(z_min_max).detach().numpy()
                
                # z = torch.randn(batch_data, data_column_dim)
                # z_minmax = (z - z.min()) / (z.max() - z.min())
                # synthetic_data = self.generator(z_minmax).detach().numpy()
                synthetic_data = scaler.inverse_transform(synthetic_data)
                synthetic_df = pd.DataFrame(synthetic_data, columns=self.data_real_dict["cols_to_study"])
                
                # Transform to classification column the target column
                synthetic_df[target_col] = synthetic_df[f"{target_col}_numeric"].apply(
                    lambda x: c2nn.inverse_categorical_interval(x, category_inter)
                )
                if hasattr(self.data_class, 'encode_target'):
                    synthetic_df, _ = self.data_class.encode_target(synthetic_df)
                else:
                    synthetic_df["target"] = synthetic_df[target_col]
                
                # ### Real data ###
                idx = np.random.randint(0, df_to_ml.shape[0], batch_data)
                df_real_sample = df_to_ml.loc[idx].reset_index(drop=True)
                
                df_real_sample[target_col] = df_real_sample[f"{target_col}_numeric"].apply(
                    lambda x: c2nn.inverse_categorical_interval(x, category_inter)
                )
                if hasattr(self.data_class, 'encode_target'):
                    df_real_sample, _ = self.data_class.encode_target(df_real_sample)
                else:
                    df_real_sample["target"] = df_real_sample[target_col]
                
                scenarios = [
                    ("train_real_test_real", df_real_sample, None, None),
                    ("train_gen_test_real", synthetic_df, df_real_sample, None),
                    ("train_realgen_test_real", df_real_sample, None, synthetic_df),
                ]
                
                df_results_list = []
                for scenario_name, train_data, test_data, join_data in scenarios:
                    results = self.train_model(
                        data_to_ml=train_data, 
                        data_test=test_data, 
                        target_col="target", 
                        ml_input_col=ml_input_list, 
                        join_data=join_data
                    )
                    rename_dict = {col: f"{col}_{scenario_name}" for col in self.metrics_skm_dict.keys()}
                    df_results_list.append(pd.DataFrame(results).rename(columns=rename_dict))
                
                help_list.append(pd.concat(df_results_list, axis=1))
            
            df_ml_metrics = pd.concat(help_list, ignore_index=True)
            
        elif self.model_name == "SDV":

            help_list = []
            for _ in range(num_generate_th):
                # ### Generated data ###
                idx = np.random.randint(0, df_to_ml.shape[0], batch_data)
                self.generator.fit(data=df_to_ml.loc[idx].reset_index(drop=True))
                synthetic_df_sdv = self.generator.sample(num_rows=batch_data)

                # Transform to classification column the target column
                synthetic_df_sdv[target_col] = synthetic_df_sdv[f"{target_col}_numeric"].apply(
                    lambda x: c2nn.inverse_categorical_interval(x, category_inter)
                )
                if hasattr(self.data_class, 'encode_target'):
                    synthetic_df_sdv, _ = self.data_class.encode_target(synthetic_df_sdv)
                else:
                    synthetic_df_sdv["target"] = synthetic_df_sdv[target_col]
                
                # ### Real data ###
                idx = np.random.randint(0, df_to_ml.shape[0], batch_data)
                df_real_sample = df_to_ml.loc[idx].reset_index(drop=True)
                
                df_real_sample[target_col] = df_real_sample[f"{target_col}_numeric"].apply(
                    lambda x: c2nn.inverse_categorical_interval(x, category_inter)
                )
                if hasattr(self.data_class, 'encode_target'):
                    df_real_sample, _ = self.data_class.encode_target(df_real_sample)
                else:
                    df_real_sample["target"] = df_real_sample[target_col]
                
                scenarios = [
                    ("train_real_test_real", df_real_sample, None, None),
                    ("train_gen_test_real", synthetic_df_sdv, df_real_sample, None),
                    ("train_realgen_test_real", df_real_sample, None, synthetic_df_sdv),
                ]
                
                df_results_list = []
                for scenario_name, train_data, test_data, join_data in scenarios:
                    results = self.train_model(
                        data_to_ml=train_data, 
                        data_test=test_data, 
                        target_col="target", 
                        ml_input_col=ml_input_list, 
                        join_data=join_data
                    )
                    rename_dict = {col: f"{col}_{scenario_name}" for col in self.metrics_skm_dict.keys()}
                    df_results_list.append(pd.DataFrame(results).rename(columns=rename_dict))
                
                help_list.append(pd.concat(df_results_list, axis=1))
            
            df_ml_metrics = pd.concat(help_list, ignore_index=True)
            
        self.eval_gen_model_dict.update(
            {"ml_train_model_comparison": df_ml_metrics}
        )
        
    def check_statistics(self, num_generate_th=30, batch_data=500):
        
        data_column_dim = self.data_train.shape[1]
        
        help_list = []
        for _ in range(num_generate_th):
            if model_dict["model_name"] in ["LinearNN", "CNN1D"]:
                scaler = MinMaxScaler()
                scaler.fit(self.data_train)
                
                z = torch.randn(batch_data, data_column_dim)
                z_min_max = (z - z.min()) / (z.max() - z.min())
                
                if self.model_name == "CNN1D":
                    z_reshape = z_min_max.reshape(batch_data, data_column_dim, 1)
                    fake_data_reshape = self.generator(z_reshape).detach().numpy()
                    synthetic_data = fake_data_reshape.reshape(batch_data, data_column_dim)
                else:
                    synthetic_data = self.generator(z_min_max).detach().numpy()
                # z = torch.randn(batch_data, data_column_dim)
                # z_minmax = (z - z.min()) / (z.max() - z.min())
                # synthetic_data = self.generator(z_minmax).detach().numpy()
                synthetic_data = scaler.inverse_transform(synthetic_data)
                synthetic_df = pd.DataFrame(
                    synthetic_data, columns=self.data_real_dict["cols_to_study"]
                )
            else:
                idx = np.random.randint(0, self.data_train.shape[0], batch_data)
                try:
                    self.generator.fit(data=self.data_train.loc[idx].reset_index(drop=True))
                except:
                    continue
                synthetic_df = self.generator.sample(num_rows=batch_data)
            
            idx = np.random.randint(0, self.data_train.shape[0], batch_data)
            df_real_sample = self.data_train.loc[idx]
            
            help_dict = {}
            for col in self.data_train.columns:
                _, p_value = sp.stats.mannwhitneyu(
                    df_real_sample[col].values, synthetic_df[col].values, 
                    alternative='two-sided'
                )
                
                help_dict[col] = p_value
            
            
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(data=data_dict["data"][data_dict["cols_to_study"]])
            quality_report = evaluate_quality(
                df_real_sample,
                synthetic_df,
                sdv_metadata
            )
            help_dict["SDV_QR_score"] = quality_report.get_score()
            
            help_list.append(help_dict)

        df_help = pd.DataFrame(help_list)   
        
        stats_dict = {}
        for col in df_help.columns:
            if col == "SDV_QR_score":
                stats_dict[f"mean_{col}"] = np.mean(df_help[col])
            else:
                stats_dict[f"up_05_pvalue_{col}"] = np.mean(df_help[col] > 0.05)
        
        df_stats = pd.DataFrame(stats_dict, index=[0])
        
        self.eval_gen_model_dict.update(
            {"check_statistics": df_stats}
        )


if __name__ == "__main__":
    all_data_list = [
        AdultDataPreprocessor,
        TitanicDataPreprocessor,
        MaternalDataPreprocessor,
        StudentDropoutDataPreprocessor,
        WineQualityDataPreprocessor,
        WisconsinDataPreprocessor,
    ]

    generative_model_list = [
        {"model_name": "SDV", "loss_function": "GaussianCopulaSynthesizer"},
        {"model_name": "SDV", "loss_function": "CopulaGANSynthesizer"},
        {"model_name": "SDV", "loss_function": "CTGANSynthesizer"},
        {"model_name": "SDV", "loss_function": "TVAESynthesizer"},
        {"model_name": "LinearNN", "loss_function": "BCELoss"},
        {"model_name": "LinearNN", "loss_function": "iqr-covmat-integral"},
        {"model_name": "LinearNN", "loss_function": "both-loss_iqr-covmat-integral"},
        {"model_name": "CNN1D", "loss_function": "BCELoss"},
        {"model_name": "CNN1D", "loss_function": "iqr-covmat-integral"},
        {"model_name": "CNN1D", "loss_function": "both-loss_iqr-covmat-integral"},
    ]
    
    metrics_skm_dict = {
        "accuracy": skm.accuracy_score,
        "kappa": skm.cohen_kappa_score,
        "f1_score": skm.f1_score,
    }

    data_model_list = [all_data_list, generative_model_list]
    comb_data_gen_list = list(itertools.product(*data_model_list))
    for data_class, model_dict in comb_data_gen_list:
        
        data_class_init = data_class()
        data_dict = data_class_init.custom_preprocess()
        
        print(data_dict["data_name"], model_dict["model_name"], model_dict["loss_function"])
        print("Columns to use model", data_dict["cols_to_study"])

        if model_dict["model_name"] != "SDV":
            gen_experiment = GeneratorExperiment(data_real_dict=data_dict, model_dict=model_dict)
            generator_model = gen_experiment.generate_model_from_data(verbose=True)
            
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)

            os.remove("SaveModels/generator_NN.pth")
            
        else:
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(data=data_dict["data"][data_dict["cols_to_study"]])
            if model_dict["loss_function"] == "GaussianCopulaSynthesizer":
                generator_model = GaussianCopulaSynthesizer(sdv_metadata)
            elif model_dict["loss_function"] == "CopulaGANSynthesizer":
                generator_model = CopulaGANSynthesizer(sdv_metadata)
            elif model_dict["loss_function"] == "CTGANSynthesizer":
                generator_model = CTGANSynthesizer(sdv_metadata)
            elif model_dict["loss_function"] == "TVAESynthesizer":
                generator_model = TVAESynthesizer(sdv_metadata)
            else:
                ValueError(model_dict["loss_function"], "not implemented")
        
        # Evaluate generator model
        eval_gen_model = EvaluateGeneratorModel(
            data_class=data_class_init,
            data_real_dict=data_dict,
            generator=generator_model,
            model_name=model_dict["model_name"],
        )
        
        try:
            eval_gen_model.check_statistics()
        except:
            pass
        
        try:
            eval_gen_model.ml_train_model_comparison()
        except:
            pass
        
        try:
            eval_gen_model.ml_difference_real_generated_data()
        except:
            pass

        generator_eval_results_dict = eval_gen_model.eval_gen_model_dict

        data_name = data_dict["data_name"]
        model_name = model_dict["model_name"]
        loss_function_n = model_dict.get("loss_function", model_name)
        save_dict_pickle = (f"Results/GenResults_{data_name}_{model_name}-{loss_function_n}.pkl")
        with open(save_dict_pickle, "wb") as f:
            pickle.dump(generator_eval_results_dict, f)
            
        del data_class
