import seaborn as sns
import matplotlib.pyplot as plt


def generate_bar_chart_from_totalfrequencies(df, filename='output.png'):
    # Defines global formatting
    sns.set(style="whitegrid", font_scale=0.7)
    sns.set_color_codes("pastel")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 16))

    # get the maximum value
    maxfreq = df.max(axis=1).astype(int)[0]
    print(maxfreq)

    # sorts columns by frequency
    df = df.sort_values(by=0, axis=1, ascending=False)

    # initialized barplot
    bp = sns.barplot(data=df, label="Total Frequency", color="b", orient='h')

    # Add a legend and informative axis label
    ax.legend(ncol=1, loc="lower right", frameon=True)
    ax.set(xlim=(0, maxfreq * 1.05), ylabel="terms",
           xlabel="frequency")

    sns.despine(left=True, bottom=True)
    plt.show()
    plt.savefig(filename)
