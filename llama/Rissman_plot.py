import matplotlib.pyplot as plt

from llama.Rissman_similarity_analysis import compare_attn_vs_gate
from llama.analogy_analyze.analogy_llama import compare_GPT_spots
from llama.carnivore_herbivore import plot_layers_cross_species
from llama.devereux_single_feature import plot_square


def plot_2rel_3b(activation_model='meta-llama/Llama-3.2-3b', bury=False):
    if bury:
        activation_model = (activation_model, 'bury')
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    plt.sca(axs[0, 0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         do_xlabel=False)
    plt.gca().text(-0.13, 1.11, 'a.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    plt.sca(axs[0, 1])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=(True, 'deve'), norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         do_xlabel=False)
    plt.gca().text(-0.13, 1.11, 'b.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    plt.sca(axs[1, 0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False)
    plt.gca().text(-0.13, 1.11, 'c.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    if not bury:
        plt.sca(axs[1, 1])
        plot_layers_cross_species(activation_model=activation_model,
                                  food_second='both')
        plt.gca().text(-0.13, 1.11, 'd.', transform=plt.gca().transAxes,
                       fontsize=18, fontweight='bold', va='center')


    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    labels = ['Residual (input)', 'Attention addition',
              'FFN addition', 'Attention weights']
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=14,
               frameon=False, columnspacing=0.8, handletextpad=0.3,
               markerscale=2, handlelength=1.5)


    #
    # for ax, letter in zip(axs, ['a.', 'b.', 'c.']):
    #     ax.text(-0.17, 1.20, letter, transform=ax.transAxes,
    #             fontsize=18, fontweight='bold', va='center')

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
    if '3b' in activation_model or '3b' in activation_model[0]:
        copies = (0, 1, 2, 3)
    else:
        copies = (0, 1)

    # if bury:
    #     fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    # else:
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    plt.sca(axs[0])
    if not bury:
        if '3b' in activation_model or '3b' in activation_model[0]:
            compare_GPT_spots(activation_model=activation_model, position=1,
                              do_r2=False, copies=copies, flip_within=False, analogy=1,
                              do_ylabel=True)
    # if bury:
    #     plt.sca(axs[0])
    # else:
    plt.sca(axs[1])
    if '3b' in activation_model or '3b' in activation_model[0]:
        compare_GPT_spots(activation_model=activation_model, position=3,
                          do_r2=False, copies=copies, flip_within=False, analogy=1,
                          do_ylabel=False)

    # if bury:
    #     plt.sca(axs[1])
    # else:
    plt.sca(axs[2])
    compare_GPT_spots(activation_model=activation_model, position=3,
                      do_r2=False, copies=copies, flip_within=True, analogy=1,
                      do_ylabel=False)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    labels = ['Residual (input)', 'Attention addition',
              'FFN addition']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14,
               frameon=False, columnspacing=0.8, handletextpad=0.3,
               markerscale=2, handlelength=1.5)
    for ax, letter in zip(axs, ['a.', 'b.', 'c.']):
        ax.text(-0.17, 1.20, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='center')

    if bury:
        plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.85, bottom=0.2,
                            hspace=0.3)
    else:

        # top = 0.765 if bury else 0.84
        # if bury:
        #     plt.suptitle('Buried texts', x=0.52)
        plt.subplots_adjust(wspace=0.25, left=0.08, right=0.96, top=0.84, bottom=0.2,
                            hspace=0.3)
    plt.show()


def plot_all_buried(activation_model='meta-llama/Llama-3.2-3b'):
    # (a.) Experiment 1: averages, (b.) Experiment 1: residual stream many features
    # (c.) Experiment 2A, (d.) Experiment 2B, (e.) Experiment 2C
    # (f.) Experiment 3B, (g.) (Experiment 3C
    plot_square(bury=True, do_legend=True, just2=True,
                suptitle='Buried semantics representation')

    activation_model_bury = (activation_model, 'bury')
    # activation_model = activation_model
    plt.rcParams.update({'font.size': 14})
    plot_exp2_triplet(activation_model_bury, letters=('c.', 'd.', 'e.'))

    if '3b' in activation_model or '3b' in activation_model[0]:
        copies = (0, 1, 2, 3)
    else:
        copies = (0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    plt.sca(axs[0])
    empty_plot(axs[2])
    plt.text(0.4, 0.55, 'N/A\n(Equivalent to g.)', ha='center', va='center', fontsize=18)
    plt.sca(axs[1])
    # if '3b' in activation_model or '3b' in activation_model[0]:
    compare_GPT_spots(activation_model=activation_model_bury, position=3,
                      do_r2=False, copies=copies, flip_within=False, analogy=1)

    # if bury:
    #     plt.sca(axs[1])
    # else:
    plt.sca(axs[2])
    compare_GPT_spots(activation_model=activation_model_bury, position=3,
                      do_r2=False, copies=copies, flip_within=True, analogy=1)

    for ax, letter in zip(axs, ['f.', 'g.', 'h.']):
        ax.text(-0.17, 1.16, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='center')

    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.85, bottom=0.11,
                        hspace=0.3)
    plt.show()

    # tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    # tolohs = zip(*tuples_lohand_lolbl)
    # handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    # labels = ['Residual (input)', 'Attention addition', 'FFN addition']
    # fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14,
    #            frameon=False, columnspacing=0.8, handletextpad=0.3,
    #            markerscale=2, handlelength=1.5)

    # plot_analogy_all(activation_model=activation_model, bury=True)

    # fig, axs = plt.subplots(1, 2, figsize=(13, 5))


def empty_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_exp2_triplet(activation_model, letters=('c.', 'd.', 'e.'),
                      drop=None):
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    plt.sca(axs[0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         do_xlabel=False)
    plt.sca(axs[1])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False,
                         do_xlabel=False)
    if drop is not None and 2 in drop:
        plt.sca(axs[2])
        empty_plot(axs[2])
        plt.text(0.4, 0.55, 'N/A\n(Not done)', ha='center', va='center', fontsize=18)
    else:
        plt.sca(axs[2])
        plot_layers_cross_species(activation_model=activation_model,
                                  food_second='both', xlabel=False)
    for ax, letter in zip(axs, letters):
        ax.text(-0.23, 1.12, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='center')
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.85, bottom=0.11,
                        hspace=0.3)
    plt.show()


def plot_70b():
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    plot_square(activation_model='meta-llama/Llama-3.3-70b-Instruct', bury=False,
                just2=True, suptitle='Llama-3.3-70b-Instruct results')

    activation_model_bury = (activation_model, 'bury')

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 4, figsize=(13, 5))
    plt.sca(axs[0])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         do_xlabel=False, four_piece=True)
    # compare_attn_vs_gate(activation_model=activation_model,
    #                      do_SchemeRep=(True, 'deve'), norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=(False, 'cont'),
    #                      do_xlabel=False, four_piece=True)
    plt.ylabel('Accuracy')
    plt.sca(axs[1])
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False,
                         do_xlabel=False, four_piece=True)
    plt.sca(axs[2])
    plot_layers_cross_species(activation_model=activation_model,
                              food_second='both', xlabel=False,
                              four_piece=True, cross_animal=True)

    plt.sca(axs[3])
    compare_GPT_spots(activation_model=activation_model,
                      position=3, do_r2=False, copies=(0, 1),
                      flip_within=True, analogy=1,
                      four_piece=True, xlabel=False)
    for ax, letter in zip(axs, ('c.', 'd.', 'e.', 'f.')):
        ax.text(-0.23, 1.11, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='center')
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.865, bottom=0.12,
                        hspace=0.3)
    plt.show()
    quit()

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    plt.sca(axs[0])
    compare_attn_vs_gate(activation_model=activation_model_bury,
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'),
                         do_xlabel=True, four_piece=True)
    plt.ylabel('Accuracy')
    plt.sca(axs[1])
    compare_attn_vs_gate(activation_model=activation_model_bury,
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False,
                         do_xlabel=True, four_piece=True)
    plt.sca(axs[2])
    compare_GPT_spots(activation_model=activation_model_bury,
                      position=3, do_r2=False, copies=(0, 1),
                      flip_within=True, analogy=1,
                      four_piece=True)
    for ax, letter in zip(axs, ('g.', 'h.', 'i.')):
        ax.text(-0.175, 1.11, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='center')
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.865, bottom=0.12,
                        hspace=0.3)
    plt.show()


if __name__ == '__main__':
    plot_70b()
    # plot_all_buried()
    # plot_2rel_3b()
    # plot_analogy_all()
    # plot_all_carn_herb()_
    # plot_2rel_3b(bury=False)
    # plot_analogy_all(bury=False)
    # plot_analogy_all(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                  bury=False)

    # plot_2rel_3b('meta-llama/Llama-3.3-70b-Instruct')
    # plot_2rel_3b('meta-llama/Llama-3.3-70b-Instruct', bury=True)

    # plot_2rel_3b(activation_model=('meta-llama/Llama-3.2-3b', 'bury'))
