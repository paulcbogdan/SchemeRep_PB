from llama.Rissman_similarity_analysis import compare_attn_vs_gate
from llama.analogy_analyze.analogy_llama import compare_GPT_spots
from llama.carnivore_herbivore import plot_layers_cross_species
import matplotlib.pyplot as plt

def plot_2rel_3b(activation_model='meta-llama/Llama-3.2-3b', bury=False):
    if bury:
        activation_model = (activation_model, 'bury')
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    plt.sca(axs[0, 0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         xlabel=False)
    plt.sca(axs[0, 1])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=(True, 'deve'), norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         xlabel=False)
    # plt.show()
    # quit()
    plt.sca(axs[1, 0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False)
    plt.sca(axs[1, 1])
    plot_layers_cross_species(activation_model=activation_model,)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    labels = ['Residual (input)', 'Attention addition',
              'FFN addition', 'Attention weights']
    legend = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=14,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5)

    top = 0.9 if bury else 0.94
    if bury:
        plt.suptitle('Buried texts')
    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.96, top=top, bottom=0.1,
                        hspace=0.275)
    plt.show()

def plot_all_carn_herb(activation_model='meta-llama/Llama-3.2-3b', bury=False):
    if bury:
        activation_model = (activation_model, 'bury')
    plot_layers_cross_species(activation_model=activation_model,
                              cross_animal=True, food_second=True,
                              plot_title=False)
    # plot_layers_cross_species(activation_model=activation_model,
    #                           cross_animal=True, food_second=False)
    # plot_layers_cross_species(activation_model=activation_model,
    #                           cross_animal=False, food_second=True)
    # plot_layers_cross_species(activation_model=activation_model,
    #                           cross_animal=False, food_second=False)

def plot_analogy_all(activation_model='meta-llama/Llama-3.2-3b', bury=False):
    if bury:
        activation_model = (activation_model, 'bury')

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    copies = (0, 1, 2, 3)

    plt.sca(axs[0])
    compare_GPT_spots(activation_model=activation_model, position=1,
                      do_r2=False, copies=copies, flip_within=False, analogy=1)
    plt.sca(axs[1])
    compare_GPT_spots(activation_model=activation_model, position=3,
                      do_r2=False, copies=copies, flip_within=False, analogy=1)
    plt.sca(axs[2])
    compare_GPT_spots(activation_model=activation_model, position=3,
                      do_r2=False, copies=copies, flip_within=True, analogy=1)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    labels = ['Residual (input)', 'Attention addition',
              'FFN addition']
    legend = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5)

    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.96, top=0.85, bottom=0.2,
                        hspace=0.3)
    plt.show()


if __name__ == '__main__':
    # plot_analogy_all(bury=True)
    # plot_all_carn_herb()
    # plot_2rel_3b(bury=True)
    plot_analogy_all(bury=True)

    # plot_2rel_3b('meta-llama/Llama-3.3-70b-Instruct')

    # plot_2rel_3b(activation_model=('meta-llama/Llama-3.2-3b', 'bury'))




