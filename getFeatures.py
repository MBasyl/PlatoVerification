
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def author_bar(plot_data, top_n=20):
    """
    Create a grouped bar plot showing the top N tokens for each author.

    Parameters:
    - token_freq_df: DataFrame with token frequencies for each author.
    - top_n: Number of top tokens to consider.

    Returns:
    - None (displays the bar plot).
    """
    top_tokens_by_author = {
        author: token_freq_df.loc[author].nsmallest(top_n)
        for author in token_freq_df.index
    }

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(top_tokens_by_author)
    # Transpose the DataFrame for better visualization
    # plot_data = plot_data.transpose()
    print(plot_data)

    # Create the grouped bar plot
    plt.figure(figsize=(14, 8))
    plot_data.plot(kind='bar',  width=0.8,  # cmap='viridis',
                   ax=plt.gca())

    plt.title(f"Top {top_n} PLAIN token 5-grams per Author")
    plt.ylabel("5-gram Token Frequency")
    plt.xticks(rotation=45),  # labels=plot_data.index)  # , ha="right",
    # ticks=plot_data.index, )
    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Place legend horizontally at the bottom
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize='8',
    #           ncol=len(plot_data.columns), fancybox=True, shadow=True)
    # plt.legend(title="Word-grams", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(pad=1)
    plt.savefig("adhoc5wPlatoONLY.png")
    plt.show()


def top_tokens_bar_plot(token_freq_df, top_n=20):
    """
    Create a grouped bar plot showing the top N tokens for each author.

    Parameters:
    - token_freq_df: DataFrame with token frequencies for each author.
    - top_n: Number of top tokens to consider.

    Returns:
    - None (displays the bar plot).
    """

    # Extract the top N tokens for each author
    top_tokens_by_author = {
        author: token_freq_df.loc[author].nlargest(top_n)
        for author in token_freq_df.index
    }

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(top_tokens_by_author)

    # Transpose the DataFrame for better visualization
    plot_data = plot_data.transpose()

    # Create the grouped bar plot
    plt.figure(figsize=(14, 8))
    plot_data.plot(kind='bar',  width=0.8,  # cmap='viridis',
                   ax=plt.gca())

    plt.title(f"Top {top_n} PLAIN token 5-grams per Author")
    plt.xlabel("Authors")
    plt.ylabel("5-gram Token Frequency")
    plt.xticks(rotation=45, ha="right")
    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Place legend horizontally at the bottom
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize='8',
    #           ncol=len(plot_data.columns), fancybox=True, shadow=True)
    plt.legend(title="Word-grams", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(pad=1)
    plt.savefig("adhoc5wPlatoONLY.png")
    plt.show()


def subset_df(df_raw):

    to_drop = ['PsPla0.91', 'PsPla0.1', 'PsPla0.7',
               'PsPla#0.25', 'Spe', 'Xec']  # 'Disputed'
    df = df_raw[~df_raw['author'].isin(to_drop)].copy()

    for index, row in df.iterrows():
        author = row['author']
        text = row['text']
        pattern = r'PsPla#0\.'
        # Check conditions and copy values
        if author == 'PsPla':
            df.at[index, 'author'] = f'Ps_{text}'
        elif re.match(pattern, author):
            perc = author.split('#')[1]
            # Concatenate 'percentage' to 'text'
            df.at[index, 'author'] = f'{text}_{perc}'
    df.drop(['text'], axis=1, inplace=True)

    return df


def subset_onlyPla(df_raw):

    to_drop = ['PsPla0.91', 'PsPla0.1', 'PsPla0.7', 'PsPla', 'Xen',
               'PsPla#0.25', 'PsPla#0.85', 'PsPla#0.5', 'Spe', 'Xec']  # 'Disputed'
    df = df_raw[~df_raw['author'].isin(to_drop)].copy()

    for index, row in df.iterrows():
        author = row['author']
        text = row['text']
        pattern = r'PsPla#0\.'
        # Check conditions and copy values
        if author == 'PsPla':
            df.at[index, 'author'] = f'Ps_{text}'
        elif re.match(pattern, author):
            perc = author.split('#')[1]
            # Concatenate 'percentage' to 'text'
            df.at[index, 'author'] = f'{text}_{perc}'
    df.drop(['author'], axis=1, inplace=True)

    return df
###


# , index_col=0)
df = pd.read_csv("adhoc5cPLAINPlato.csv")
# small_df = subset_df(df)
# small_df = subset_onlyPla(df)
# token_freq_df = small_df.groupby('text').mean().reset_index()
# token_freq_df.set_index('text', inplace=True)
# top_tokens_bar_plot(token_freq_df, top_n=20)

df_plato = df[df['author'].isin(['Pla'])]

df_plato.drop(['text'], axis=1, inplace=True)
token_freq_df = df_plato.groupby('author').mean().reset_index()
token_freq_df.set_index('author', inplace=True)
author_bar(token_freq_df, 50)
