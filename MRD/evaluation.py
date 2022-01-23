import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
from dython.nominal import associations


class Evaluator:
    #Creamos una classe para evaluar 
    def __init__(self, real: pd.DataFrame, fake: pd.DataFrame, cat_cols=None,n_samples=None,unique_thresh=0):
        
        self.unique_thresh = unique_thresh
        self.real = real.copy()
        self.fake = fake.copy()
        
        if cat_cols is None:
            real = real.infer_objects()
            fake = fake.infer_objects()
            self.numerical_columns = [column for column in real._get_numeric_data().columns if
                                      len(real[column].unique()) > unique_thresh]
            self.categorical_columns = [column for column in real.columns if column not in self.numerical_columns]
        else:
            self.categorical_columns = cat_cols
            self.numerical_columns = [column for column in real.columns if column not in cat_cols]

         # Make sure the number of samples is equal in both datasets.
        if n_samples is None:
            self.n_samples = min(len(self.real), len(self.fake))
        elif len(fake) >= n_samples and len(real) >= n_samples:
            self.n_samples = n_samples
        else:
            raise Exception(f'Make sure n_samples < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}')

        self.real = self.real.sample(self.n_samples)
        self.fake = self.fake.sample(self.n_samples)
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

        self.real.loc[:, self.categorical_columns] = self.real.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)
        self.fake.loc[:, self.categorical_columns] = self.fake.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)

        self.real.loc[:, self.numerical_columns] = self.real.loc[:, self.numerical_columns].fillna(
            self.real[self.numerical_columns].mean())
        self.fake.loc[:, self.numerical_columns] = self.fake.loc[:, self.numerical_columns].fillna(
            self.fake[self.numerical_columns].mean())
    
    def plot_mean_std(self, fname=None, ax=None):
        
        #Plot the means and standard deviations of each dataset. 
    
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

        ax[0].grid(True)
        ax[1].grid(True)
        real = self.real._get_numeric_data()
        fake = self.fake._get_numeric_data()
        real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
        fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
        min_mean = min(real_mean) - 1
        max_mean = max(real_mean) + 1
        line = np.arange(min_mean, max_mean)
        sns.lineplot(x=line, y=line, ax=ax[0])
        sns.scatterplot(x=real_mean,y=fake_mean,ax=ax[0])
        ax[0].set_title('Means of real and fake data')
        ax[0].set_xlabel('real data mean (log)')
        ax[0].set_ylabel('fake data mean (log)')

        real_std = np.log(np.add(real.std().values, 1e-5))
        fake_std = np.log(np.add(fake.std().values, 1e-5))
        min_std = min(real_std) - 1
        max_std = max(real_std) + 1
        line = np.arange(min_std, max_std)
        sns.lineplot(x=line, y=line, ax=ax[1])
        sns.scatterplot(x=real_std,y=fake_std,ax=ax[1])
        ax[1].set_title('Stds of real and fake data')
        ax[1].set_xlabel('real data std (log)')
        ax[1].set_ylabel('fake data std (log)')
        plt.show()
   
    def plot_distributions(self, nr_cols=3):
        """
        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.
        :param fname: If not none, saves the plot with this file name. 
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            if col not in self.categorical_columns:
                plot_df = pd.DataFrame({col: self.real[col].append(self.fake[col]), 'kind': ['real'] * self.n_samples + ['fake'] * self.n_samples})
                fig = sns.histplot(plot_df, x=col, hue='kind', ax=axes[i], stat='probability', legend=True)
                axes[i].set_autoscaley_on(True)
            else:
                real = self.real.copy()
                fake = self.fake.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Fake'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i], saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        plt.show()

    def plot_correlation_difference( self , plot_diff = True , annot=False):
    
        assert isinstance(self.real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
        assert isinstance(self.fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if self.categorical_columns is None:
            cat_cols = self.real.select_dtypes(['object', 'category'])
        if plot_diff:
            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

        real_corr = associations(self.real, nominal_columns=self.categorical_columns, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[0], cmap=cmap)['corr']
        fake_corr = associations(self.fake, nominal_columns=self.categorical_columns, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[1], cmap=cmap)['corr']

        if plot_diff:
            diff = abs(real_corr - fake_corr)
            sns.set(style="white")
            sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

        titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
        for i, label in enumerate(titles):
            title_font = {'size': '18'}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()

        plt.show()

    def plot_cumsums(self, nr_cols=4, fname=None):
    
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        plt.show()


def cdf(data_r, data_f, xlabel: str = 'Values', ylabel: str = 'Cumulative Sum', ax=None):
   
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    import matplotlib.ticker as mticker

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
        ticks_loc = ax.get_xticks()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(data_r.sort_values().unique(), rotation='vertical')

    if ax is None:
        plt.show()