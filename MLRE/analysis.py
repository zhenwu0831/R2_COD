import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict as ddict, Counter
import matplotlib.patches as patches
from helper import *

sns.set_style("whitegrid")
sns.set_palette("colorblind")  # Use colorblind-friendly palette


def analyse_results():

    if args.dataset == 'redfm':
        langs = ['de', 'en', 'es', 'fr', 'it']
    elif args.dataset == 'indore':
        langs = ['en', 'hi', 'te']

    results_df = ddict(list)

    all_lang   = ddict(list)

    for lang in langs:
        results_df['lang'].append(lang)
        for setting in ['text', 'graph', 'both']:
            f1_list = []

            for seed in [11737, 98105, 15123]:
                file = f'../predictions/{args.dataset}/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_{args.connection}-setting_{setting}-gnn-depth_2-seed_{seed}.csv'
                df = pd.read_csv(file)
                y_pred = df['pred_rel'].values
                y_true = df['true_rel'].values
                
                f1_val  = f1_score(y_true, y_pred, average='macro')

                f1_list.append(f1_val)
                all_lang[setting].append(f1_val)

            
            mean_f1 = np.mean(f1_list)
            std_f1  = np.std(f1_list)

            results_df[f'{setting}_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

            print(f'{lang} {setting} f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')
            
        for setting in ['both']:
            f1_list = []

            for seed in [11737, 98105, 15123]:
                file = f'../predictions/{args.dataset}/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_residual-setting_{setting}-gnn-depth_2-seed_{seed}.csv'
                df = pd.read_csv(file)
                y_pred = df['pred_rel'].values
                y_true = df['true_rel'].values
                
                f1_val  = f1_score(y_true, y_pred, average='macro')

                f1_list.append(f1_val)
                all_lang['residual'].append(f1_val)

            
            mean_f1 = np.mean(f1_list)
            std_f1  = np.std(f1_list)

            results_df[f'residual_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

            print(f'{lang} residual f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

        
        for setting in ['both']:
            f1_list = []

            for seed in [11737, 98105, 15123]:
                file = f'../predictions/{args.dataset}/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_{args.connection}-setting_{setting}-gnn-depth_2-head_dim_2048-seed_{seed}.csv'
                df = pd.read_csv(file)
                y_pred = df['pred_rel'].values
                y_true = df['true_rel'].values
                
                f1_val  = f1_score(y_true, y_pred, average='macro')

                f1_list.append(f1_val)
                # all_lang['mod_loss'].append(f1_val)

            f1_list_sorted = sorted(f1_list)[1:]  #          

            mean_f1 = np.mean(f1_list_sorted)
            std_f1  = np.std(f1_list_sorted)

            all_lang['mod_loss'].extend(f1_list_sorted)

            results_df[f'mod_loss_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

            print(f'{lang} mod_loss f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')




    results_df['lang'].append('ALL')
    for setting in ['text', 'graph', 'both', 'residual', 'mod_loss']:        

        mean_f1 = np.mean(all_lang[setting])
        std_f1  = np.std(all_lang[setting])

        results_df[f'{setting}_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

        print(f'ALL {setting} f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')
    
    



    



    print(results_df)
    results_df = pd.DataFrame(results_df)

    

    results_df.to_csv(f'../results/{args.dataset}_results_{args.connection}.csv', index=False)



def project_complementarity():

    if args.dataset == 'redfm':
        langs = ['de', 'en', 'es', 'fr', 'it']
    elif args.dataset == 'indore':
        langs = ['en', 'hi', 'te']

    
    for lang in langs:
        datawise_results = ddict(list)

        idxwise_results = ddict(lambda: ddict(lambda: ddict(int)))
        true_results    = []

        for setting in ['text', 'graph']:    

            for seed in [11737, 98105, 15123]:
                file = f'../predictions/{args.dataset}/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_{args.connection}-setting_{setting}-gnn-depth_2-seed_{seed}.csv'
                df = pd.read_csv(file)

                for idx, row in df.iterrows():
                    idxwise_results[idx][setting][row['pred_rel']] += 1

                    if len(true_results) <= idx:  # If we haven't seen this index before
                        true_results.append(row['true_rel'])
                    else:
                        assert true_results[idx] == row['true_rel'], "Mismatch in true relation"


        for idx in idxwise_results.keys():
            text_counts     = idxwise_results[idx]['text']
            graph_counts    = idxwise_results[idx]['graph']

            # Determine the most common prediction for each setting
            text_pred = max(text_counts, key=text_counts.get)
            graph_pred = max(graph_counts, key=graph_counts.get)

            # Get the true relation for this index
            true_rel = true_results[idx]
            
            text_rel_acc = 0
            graph_rel_acc = 0

            if text_pred == true_rel:
                text_rel_acc = 1
            if graph_pred == true_rel:
                graph_rel_acc = 1

            if text_rel_acc == 1 and  graph_rel_acc == 0 :
                datawise_results['agreement'].append('text')
            elif text_rel_acc == 0 and graph_rel_acc == 1:
                datawise_results['agreement'].append('graph')
            elif text_rel_acc == 0 and graph_rel_acc == 0:
                datawise_results['agreement'].append('none')
            elif text_rel_acc == 1 and graph_rel_acc == 1:
                # Both predictions are correct, we can consider it as agreement
                datawise_results['agreement'].append('both')
            
        # Sample data
        classes         = list(datawise_results['agreement'])  # Copy the list to avoid modifying the original

        

        class_mapping   = {'text': 'red', 'graph': 'blue', 'none': 'gray', 'both': 'green'}  # Define a mapping for the colors
        colors          = [class_mapping[c] for c in classes]  # Convert class labels to colors

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 1.5))

        # Draw colored rectangles manually to avoid gaps
        for i, color in enumerate(colors):
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, color=color))  # (x, y), width, height

        # Formatting
        ax.set_xlim(0, len(classes))  # Set x-axis limits
        ax.set_ylim(0, 1)  # Keep the height consistent
        ax.set_yticks([])  # Hide y-axis
        # ax.set_xticks(range(len(classes)))  # Set x-ticks at each index
        # ax.set_xticklabels(range(1, len(classes) + 1), fontsize=8)  # Show index labels

        # Remove axis borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Labels and title
        # ax.set_xlabel("Index")
        ax.set_title(f"Language: {lang}")

        # add the legend to the plot
        legend_labels = {
            'red': 'Text Only',
            'blue': 'Graph Only',
            'gray': 'Neither Text nor Graph',
            'green': 'Both Text and Graph'
        }

        # plot legend
        legend_patches = [patches.Patch(color=color, label=label) for color, label in legend_labels.items()]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

        plt.show()
        plt.savefig(f'../results/{args.dataset}_{lang}_complementarity_{args.connection}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

def get_statistics():

    if args.dataset == 'redfm':
        langs = ['de', 'en', 'es', 'fr', 'it']
    
    args.data_dir										=   f'../data/{args.dataset}'
    
    for lang in langs:
        src_file 											= 	f'{args.data_dir}/{lang}_mbert-base_stanza_combined.dill'

        src_dataset										    =   load_dill(src_file)

        train_data, dev_data, test_data     			    =   src_dataset['train'], src_dataset['validation'], src_dataset['test']

        labels                                              =   []

        for elem in train_data:
            label_arr = elem['label']
            labels.append(np.where(label_arr==1)[0][0])
            
            
        print(f'{lang}\t{len(train_data)}\t{len(dev_data)}\t{len(test_data)}\t{len(set(labels))}')
    


def process_results():

    df = pd.read_csv('results.csv')

    langs       = ['de', 'en', 'es', 'fr', 'it']

    # connections = ['mulco', 'residual', 'mulco_combined']

    # settings    = ['text', 'graph', 'both']

    keys          = {
        'mulco': ['text', 'graph', 'both'],
        'residual': ['both'],
        'mulco_combined': ['graph', 'both']
    }

    
    results_df = ddict(list)

    for lang in langs:

        results_df['lang'].append(lang)

        for key in keys:

            for val in keys[key]:            

                curr_df             = df[(df['src_lang'] == lang) & (df['connection'] == key) & (df['setting'] == val)]
                f1_list             = curr_df['test_f1'].values

                f1_list             = sorted(f1_list)[1:]  # Remove the worst run

                mean_f1             = np.mean(f1_list)
                std_f1              = np.std(f1_list)

                results_df[f'{key}_{val}_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

                print(f'{lang} {key} {val} f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')
                

    results_df['lang'].append('ALL')
    for key in keys:
        for val in keys[key]:
            curr_df             = df[(df['connection'] == key) & (df['setting'] == val)]
            f1_list             = curr_df['test_f1'].values
            mean_f1             = np.mean(f1_list)
            std_f1              = np.std(f1_list)
            results_df[f'{key}_{val}_f1'].append(f'{np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')

            print(f'ALL {key} {val} f1: {np.round(100*mean_f1,2)}\pm{np.round(100*std_f1,2)}')
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('results/f1_results_sorted.csv', index=False)


def plot_mulco_visualizations():

    langs       = ['de', 'en', 'es', 'fr', 'it']
    seeds       = [11737, 98105, 15123]   

    viz_dict    = ddict(list)

    for lang in langs:

        for seed in seeds:

            epoch_dir =f'/data/shire/projects/mulco_multilingual_re/epochwise_images/redfm/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_mulco-setting_both-gnn-depth_2-head_dim_2048-seed_{seed}/'

            epochs = ['None'] + list(range(0, 50))

            for epoch in epochs:

                try:
                    
                    train_data = json.load(open(f'{epoch_dir}/train_results_dict_epoch_{epoch}.json'))
                    dev_data   = json.load(open(f'{epoch_dir}/dev_results_dict_epoch_{epoch}.json'))

                    # print(train_data, dev_data)
                    if epoch == 'None':
                        epoch_val = 0
                    else:
                        epoch_val = int(epoch)+1

                    for metric in train_data:
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('train')

                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)
                        viz_dict['value'].append(train_data[metric])


                    for metric in dev_data:    
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('dev')
                        
                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)

                        viz_dict['value'].append(dev_data[metric])
                except Exception as e:
                    # print(f'Error processing {lang} seed {seed} epoch {epoch}: {e}')
                    continue

    viz_df = pd.DataFrame(viz_dict)
    viz_df.to_csv('../results/viz_df.csv', index=False)

    # how to plot the results of viz_df

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")  # Use colorblind-friendly palette
    sns.set_context("paper", font_scale=1.2)

    # plot the relplot for different metric

    g= sns.relplot(data=viz_df, x='epoch', y='value', hue='lang', col='metric', row='split', kind='line', facet_kws={'sharey': False, 'sharex': True}, height=3, aspect=1.5, errorbar=None)
    g.set(ylim=(0, 1))
    plt.savefig('../results/mulco_epochwise_metrics.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_residual_visualizations():

    langs       = ['de', 'en', 'es', 'fr', 'it']
    seeds       = [11737, 98105, 15123]   

    viz_dict    = ddict(list)

    for lang in langs:

        for seed in seeds:

            epoch_dir =f'/data/shire/projects/mulco_multilingual_re/epochwise_images/redfm/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_residual-setting_both-gnn-depth_2-head_dim_2048-seed_{seed}/'

            epochs = ['None'] + list(range(0, 50))

            for epoch in epochs:

                try:
                    
                    train_data = json.load(open(f'{epoch_dir}/train_results_dict_epoch_{epoch}.json'))
                    dev_data   = json.load(open(f'{epoch_dir}/dev_results_dict_epoch_{epoch}.json'))

                    # print(train_data, dev_data)
                    if epoch == 'None':
                        epoch_val = 0
                    else:
                        epoch_val = int(epoch)+1

                    for metric in train_data:
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('train')

                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)
                        viz_dict['value'].append(train_data[metric])


                    for metric in dev_data:    
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('dev')
                        
                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)

                        viz_dict['value'].append(dev_data[metric])
                except Exception as e:
                    # print(f'Error processing {lang} seed {seed} epoch {epoch}: {e}')
                    continue

    viz_df = pd.DataFrame(viz_dict)
    viz_df.to_csv('../results/res_viz_df.csv', index=False)

    # how to plot the results of viz_df

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")  # Use colorblind-friendly palette
    sns.set_context("paper", font_scale=1.2)

    # plot the relplot for different metric

    g=sns.relplot(data=viz_df, x='epoch', y='value', hue='lang', col='metric', row='split', kind='line', facet_kws={'sharey': False, 'sharex': True}, height=3, aspect=1.5, errorbar=None)
    g.set(ylim=(0, 1))
    plt.savefig('../results/residual_epochwise_metrics.png', bbox_inches='tight')
    plt.clf()
    plt.close()



def plot_mulco_combined_visualizations():

    langs       = ['de', 'en', 'es', 'fr', 'it']
    seeds       = [11737, 98105, 15123]   

    viz_dict    = ddict(list)

    for lang in langs:

        for seed in seeds:

            epoch_dir =f'/data/shire/projects/mulco_multilingual_re/epochwise_images/redfm/{lang}_{lang}-model_mbert-base-parser_stanza-gnn_rgcn-connection_mulco_combined-setting_both-gnn-depth_2-head_dim_2048-seed_{seed}/'

            epochs = ['None'] + list(range(0, 50))

            for epoch in epochs:

                try:
                    
                    train_data = json.load(open(f'{epoch_dir}/train_results_dict_epoch_{epoch}.json'))
                    dev_data   = json.load(open(f'{epoch_dir}/dev_results_dict_epoch_{epoch}.json'))

                    # print(train_data, dev_data)
                    if epoch == 'None':
                        epoch_val = 0
                    else:
                        epoch_val = int(epoch)+1

                    for metric in train_data:
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('train')

                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)
                        viz_dict['value'].append(train_data[metric])


                    for metric in dev_data:    
                        viz_dict['lang'].append(lang)
                        viz_dict['epoch'].append(epoch_val)
                        viz_dict['split'].append('dev')
                        
                        if metric == 'within_group1':
                            viz_dict['metric'].append('within text')
                        elif metric == 'within_group2':
                            viz_dict['metric'].append('within graph')
                        else:
                            viz_dict['metric'].append(metric)

                        viz_dict['value'].append(dev_data[metric])
                except Exception as e:
                    # print(f'Error processing {lang} seed {seed} epoch {epoch}: {e}')
                    continue

    viz_df = pd.DataFrame(viz_dict)
    viz_df.to_csv('../results/viz_df.csv', index=False)

    # how to plot the results of viz_df

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")  # Use colorblind-friendly palette
    sns.set_context("paper", font_scale=1.2)

    # plot the relplot for different metric

    g= sns.relplot(data=viz_df, x='epoch', y='value', hue='lang', col='metric', row='split', kind='line', facet_kws={'sharey': False, 'sharex': True}, height=3, aspect=1.5, errorbar=None)
    g.set(ylim=(0, 1))
    plt.savefig('../results/mulco_combined_epochwise_metrics.png', bbox_inches='tight')
    plt.clf()
    plt.close()



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--step',        type=str, required=True)
    argparser.add_argument('--dataset',     type=str, default='redfm')
    argparser.add_argument('--connection',  type=str, default='mulco_combined')


    args = argparser.parse_args()

    if args.step == 'analyse':
        analyse_results()
    elif args.step == 'project':
        project_complementarity()

    elif args.step == 'stats':

        get_statistics()
    
    elif args.step == 'results':
        process_results()
    
    elif args.step == 'mulco_viz':
        plot_mulco_visualizations()

    elif args.step == 'res_viz':
        plot_residual_visualizations()
    
    elif args.step == 'combined_viz':
        plot_mulco_combined_visualizations()
