import seaborn as sns
import matplotlib.pyplot as plt

def plot_panel(data, x, y, col, title, kind='line'):
    g = sns.relplot(
        data=data, 
        x=x, 
        y=y, 
        col=col,      # Create a different graph for each store
        kind=kind,         # Specify line plot
        col_wrap=3,          # Wraps to a new row after 3 graphs
        height=4,            # Height of each individual panel
        aspect=1.2          # Width-to-height ratio of each panel
    )

    # Add a main title for the entire panel
    g.fig.suptitle(title, 
                fontsize=16, fontweight='bold', y=1.05)
    g.set_axis_labels(f'Day'.format(), f'{' '.join(title.split()[:2])}'.format(y))

    plt.show();
    plt.close();

    return g


if __name__ == '__main__':
    pass