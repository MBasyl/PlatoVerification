import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def match_trigrams_to_full_names(trigram_names, full_names):
    matched_names = []

    for trigram in trigram_names:
        # Check if the trigram matches the first three characters of any full name
        matches = [full_name for full_name in full_names if full_name.lower(
        ).startswith(trigram.lower())]

        if matches:
            matched_names.extend(matches)

    return matched_names


def word_bar_chart(data, x_label, y_label, title, x='Texts', y='Word_count', color='Author'):
    x_value = data[x]
    y_value = data[y]
    # Plotting
    plt.figure(figsize=(10, 6))

    fig = px.bar(data, x=x_value, color=color,
                 y=y_value,
                 title=title,
                 barmode='group', text_auto=True
                 )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.show()


def text_bar_chart(data,  title, x='Authors', y='Word count', color='Label'):
    x_value = data[x]
    y_value = data[y]
    # Plotting
    plt.figure(figsize=(10, 6))

    fig = px.bar(data, x=x_value, color=color,
                 y=y_value,
                 title=title,
                 # barmode='overlay',  # relative
                 text_auto=True
                 )

    fig.show()


def reduce_sparsity(df, threshold=0.25):
    """
    Reduce sparsity in a DataFrame at the column level.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): Threshold for sparsity. Columns with a fraction
                         of zero values above this threshold will be removed.

    Returns:
    - DataFrame with reduced sparsity.
    """
    # Calculate the fraction of zero values per column
    zero_fraction = (df == 0).mean()

    # Identify columns to keep based on the threshold
    columns_to_keep = zero_fraction[zero_fraction <= threshold].index

    # Create a new DataFrame with reduced sparsity
    reduced_df = df[columns_to_keep].copy()

    return reduced_df


if __name__ == "__main__":

    df = pd.read_csv(
        "frequency_matrixes/frequency('w', 5)PARSEDPlato.csv", index_col=0)
    print(df)
    # Apply the function to reduce sparsity
    reduced_df = reduce_sparsity(df, threshold=0.75)

    # Display the reduced DataFrame
    print(reduced_df)
    exit(0)
    df = pd.read_csv("OverviewrawCorpus2.csv", sep=";")
    # Get full names of authors:
    f = pd.read_csv("authorListrev.txt", header=None)
    authorList = set(f.iloc[:, 0].tolist())  # 13

    trigrams = df.Text.str.split("_").str[0].to_list()
    df['Authors'] = match_trigrams_to_full_names(trigrams, authorList)
    df['Text'] = df.Text.str.split("_").str[1]

    text_bar_chart(df, title='Words per Author', x='Authors',
                   y='Word count', color='Label')

    # plot_bar_chart(data=word_data, x_label='Authors/Works', y_label='Number of Words', title='Words Overview')

    # Plotting the number of words for each work
    # plot_bar_chart(data=df.set_index('Text_id'), x_label='Works',
    # y_label='Number of Words', title='Words per Work')
