import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from scipy import stats

# visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('seaborn-v0_8-whitegrid')

import seaborn as sns
sns.set_theme(style="darkgrid")

"""
Imports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
torch.set_float32_matmul_precision('medium')


"""
Visualizations
"""


def create_subplots(nrows, ncols, width_per_col, height_per_row):
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols * width_per_col, nrows * height_per_row)  # width, height
    )

    return fig, axs


def draw_box_plots(df: pd.DataFrame, target, width_per_col, height_per_row):
    nrows = len(df.select_dtypes(include=['number']).columns)
    fig, axs = create_subplots(nrows, 1, width_per_col, height_per_row)
    plt.subplots_adjust(wspace=0.5, hspace=0.9, top=0.9, bottom=0.1)

    y_name = target.replace('_', ' ').title()
    # plt.suptitle(f'{y_name} vs Numeric Variables', size=18, y=0.9)

    from pypalettes import get_hex
    n_categories = int(df.nunique()[target])
    palette = get_hex("Acadia", keep_first_n=n_categories)

    columns = df.select_dtypes(include=['number']).columns.sort_values()

    for index, column in enumerate(columns):
        b = sns.boxplot(
            data=df,
            y=target, x=column,
            ax=axs[index],
            palette=palette,
            width=0.6
        )

        x_name = column.replace('_', ' ').title()

        b.axes.set_title(f"{y_name} vs {x_name}", fontsize=16, pad=20)
        b.set_xlabel(x_name, fontsize=13)
        b.set_ylabel(y_name, fontsize=13)

        b.tick_params(labelsize=13)


def plot_count_grid(data, columns_info, grid_size=(2, 2), font_size=14, tick_label_size=12, figsize=(16, 12), hue_column=None, wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1):
    """
    Generalized function to plot a grid of count plots with adjustable padding and font sizes.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to plot.
    - columns_info (list of tuples): Each tuple should contain (column_name, plot_title, x_label).
    - grid_size (tuple): Number of rows and columns in the grid (default is (2, 2)).
    - font_size (int): Font size for titles (default is 14).
    - tick_label_size (int): Font size for tick labels (default is 12).
    - figsize (tuple): Size of the entire figure (default is (16, 12)).
    - hue_column (str): The column used for hue, typically for showing categories like 'Survived'.
    - wspace (float): The amount of width reserved for blank space between subplots (default is 0.3).
    - hspace (float): The amount of height reserved for blank space between subplots (default is 0.3).
    - left, right, top, bottom (float): Padding around the entire figure (default values add padding).
    """

    # Set up the figure and axes based on grid size
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)

    # Increase font size globally
    plt.rc('axes', titlesize=font_size)  # Title font size
    plt.rc('axes', labelsize=font_size)  # Label font size
    plt.rc('xtick', labelsize=tick_label_size)  # X-tick label font size
    plt.rc('ytick', labelsize=tick_label_size)  # Y-tick label font size
    plt.rc('legend', fontsize=font_size)  # Legend font size

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through each subplot
    for ax, (column, title, xlabel) in zip(axes, columns_info):
        sns.countplot(x=column, hue=hue_column, data=data, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')

    # Adjust layout to prevent overlap and add padding
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, top=top, bottom=bottom)
    plt.show()


def draw_scatter_plot(df, x, y):
    plt.figure(figsize=(15, 6))
    plt.scatter(x=x, y=y, data=df, color='crimson', alpha=0.4)
    plt.xlabel(x, weight='bold')
    plt.ylabel(y, weight='bold')
    plt.show()


def three_chart_plot(df, feature):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    grid = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')

    sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1, fit=stats.norm)
    plt.axvline(x=df[feature].mean(), c='red')
    plt.axvline(x=df[feature].median(), c='green')

    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:, feature], plot=ax2)

    # Customizing the Box Plot
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3)

    plt.show()


def two_chart_homoscedasticity_plot(df, x, y):
    fig = plt.figure(constrained_layout=True, figsize=(15, 7))
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title('Scatter Plot', fontsize=18)
    sns.scatterplot(data=df, x=x, y=y, ax=ax1, color='crimson', alpha=0.4)
    sns.regplot(x=df[x], y=df[y], ax=ax1)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_title('Residual Plot', fontsize=18)
    sns.residplot(x=df[x], y=df[y], ax=ax2)

    plt.show()


"""
Correlations
"""


def corr_against_numeric_target(df, target):
    corr_num_col = df.corr(numeric_only=True)[target].sort_values(ascending=False).to_frame()

    f, ax = plt.subplots(figsize=(30, 1))

    sns.heatmap(corr_num_col.T, annot=True, fmt='.2f', annot_kws={'size': 14})

    plt.title('Correlation of Numerical Features', fontsize=18, pad=25)
    plt.xticks(rotation=85, fontsize=14)
    plt.yticks(color='dodgerblue')

    plt.show()

    return corr_num_col


def corr_among_numerics(df):
    f, ax = plt.subplots(figsize=(8, 6))

    corr_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        square=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 14},
        cmap="BuPu"
    )

    plt.title('Correlation among Numerical Features', fontsize=16, pad=25)
    plt.xticks(rotation=25, fontsize=15)
    plt.yticks(rotation=25, fontsize=15)

    plt.show()


def corr_against_category_target(df, target):
    # The binary categorical variable
    category = df[target]
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = numeric_columns.drop('category', errors='ignore')

    correlation_results = []

    from scipy.stats import pointbiserialr

    for column in numeric_columns:
        correlation, p_value = pointbiserialr(df[column], category)
        correlation_results.append({
            'Numeric Variables': column,
            'Correlation': correlation,
            'P-Value': p_value
        })

    correlation_results = pd.DataFrame(correlation_results)
    correlation_results = correlation_results.sort_values(by=['Correlation'], ascending=False)
    correlation_results = correlation_results.reset_index(drop=True)
    return correlation_results


"""
Skewness
"""


def fix_skewness_yeojohnson(df, skew_threshold=0.5):
    from scipy.stats import yeojohnson

    # Copy the DataFrame to avoid modifying the original
    df_fixed = df.copy()

    # Identify numeric columns
    numeric_columns = df_fixed.select_dtypes(include=['number']).columns

    # Apply Yeo-Johnson transformation
    for column in numeric_columns:
        skewness = df_fixed[column].skew()
        if abs(skewness) > skew_threshold:
            # print(f"Column '{column}' has skewness of {skewness:.2f}. Applying Yeo-Johnson transformation.")

            # Apply Yeo-Johnson transformation
            df_fixed[column], _ = yeojohnson(df_fixed[column])

    return df_fixed


"""
Missing Values
"""


def null_table(data):
    """
    A function which returns the number and percentage of null values in the given dataset.
    """
    indices = data.isnull().sum().index
    values = data.isnull().sum().values
    percentages = []

    for i in indices:
        percentages.append((data[i].isnull().sum() / data[i].shape[0]) * 100)

    d = {'Columns': indices,
         'Null Count': values,
         'Percentage': percentages}

    null_frame = pd.DataFrame(data=d)
    null_frame = null_frame.sort_values(
        by='Percentage',
        ascending=False
    ).reset_index(drop=True).query('Percentage > 0')

    return null_frame


def msv_bar_plot(data, thresh=20, color='black', edgecolor='black', height=3, width=15):
    plt.figure(figsize=(width, height))
    percentage = (data.isnull().mean()) * 100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')

    plt.title('Missing values percentage per column', fontsize=20, pad=15)

    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh + 12.5,
             f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
             ha='left', va='top')
    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh - 5, f'Columns with less than {thresh}% missing values',
             fontsize=12, color='green',
             ha='left', va='top')

    plt.xlabel('Columns', size=15)
    plt.ylabel('Missing values percentage', size=15)
    plt.yticks(weight='bold')

    plt.show()


"""
Plot Metrics
"""


def plot_validation_curve(X, y, model, param_name, param_range):
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(model,
                                                 X, y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=5,
                                                 n_jobs=-1
                                                 )

    plt.figure()
    plt.plot(param_range, np.mean(train_scores, axis=1), label='Training score', color='blue')
    plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation score', color='red')
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(f'Validation Curve for {model.__class__.__name__}', fontsize=18)
    plt.legend(loc='best')
    plt.show()


def plot_learning_curve(X, y, model):
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X, y,
                                                            cv=5,
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10)
                                                            )

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score', color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score', color='red')
    plt.xlabel('Training examples', fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(f'Learning Curve for {model.__class__.__name__}', fontsize=18)
    plt.legend(loc='best')
    plt.show()


def plot_confusion_matrix(y_test, y_pred, cmap='Blues', text_size=22):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Customize the display
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=cmap, ax=ax, colorbar=False)

    # Add colorbar with customized location and size
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)

    # Titles and labels
    plt.title('Confusion Matrix', fontsize=18, pad=30)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Increase the size of numbers inside the matrix and remove unwanted formatting
    for text in ax.texts:
        text.set_fontsize(text_size)

    # Show the plot with a tighter layout
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, is_binary=True):
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.figure()

    if is_binary:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        plt.plot(fpr, tpr,
                 color='blue',
                 lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) if y_test.ndim == 1 else y_test
        n_classes = y_test_bin.shape[1] if y_test_bin.ndim > 1 else 2

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_test, y_pred_proba, is_binary=True):
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure()

    if is_binary:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        plt.plot(recall, precision,
                 color='blue',
                 lw=2,
                 label=f'Precision-Recall curve (AP = {avg_precision:.2f})')

    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) if y_test.ndim == 1 else y_test
        n_classes = y_test_bin.shape[1] if y_test_bin.ndim > 1 else 2

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision,
                     lw=2,
                     label=f'Class {i} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title('Precision-Recall Curve', fontsize=18)
    plt.legend(loc="lower left")
    plt.show()


"""
Images
"""

def show_grid(images, labels=None, label_fontsize=15, figsize_factor=2, rows=None, cols=None):
    # If only one image is passed, convert it to a list
    if not isinstance(images, list):
        images = [images]

    # Calculate rows and columns based on provided values
    if rows is None and cols is None:
        raise ValueError("Either rows or cols must be specified")

    if rows is None:
        rows = (len(images) + cols - 1) // cols  # Calculate rows if cols is provided

    if cols is None:
        cols = (len(images) + rows - 1) // rows  # Calculate cols if rows is provided

    # Create a matplotlib figure with subplots in a grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figsize_factor, rows * figsize_factor), squeeze=False)

    for i, img in enumerate(images):
        img = img.detach()  # Detach the tensor from the computation graph, if needed
        img = TF.to_pil_image(img)  # Convert to PIL image

        # Get row and column indices for the grid
        row, col = divmod(i, cols)

        # Display the image
        axs[row, col].imshow(np.asarray(img))

        # Set the title if labels are provided
        if labels:
            axs[row, col].set_title(labels[i], fontsize=label_fontsize)

        # Remove axis labels and ticks
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # Hide any empty subplots if there are fewer images than grid slots
    for i in range(len(images), rows * cols):
        fig.delaxes(axs.flatten()[i])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


