import argparse

from seq2seq.attributions import get_attention_attributions, get_IG_attributions, plot_attributions, load_model

# python attribute.py --expt_dir experiment/2020_01_07_11_15_29 --load_checkpoint 4_8769 --attr_type attention --src_seq "the british government is doing a good job" --verbose --output_fig_path example.png


def separator():
    print('-'*100)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True)
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', required=True)
    parser.add_argument('--src_seq', action='store', dest='src_seq', required=True)
    parser.add_argument('--attr_type', action='store', dest='attr_type', required=True, choices=['IG', 'attention'])
    parser.add_argument('--no_display', action='store_true', default=False)
    parser.add_argument('--output_fig_path', help='path to output figure', default=None)                     
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--fig_title', default=None)
    parser.add_argument('--fig_size', nargs='+', help='Size of figure to be plotted', default=(10,6))
    opt = parser.parse_args()
    return opt

    
    
def main(opt):
    model, src_vocab, tgt_vocab = load_model(opt)
    src_seq = opt.src_seq.lower().split(' ')
    
    if opt.verbose:
        print('Input sequence:', src_seq)
        print()
    
    if opt.attr_type=='attention':
        output_seq, scores = get_attention_attributions(src_seq, model, src_vocab, tgt_vocab)
    elif opt.attr_type=='IG':
        output_seq, scores = get_IG_attributions(src_seq, model, src_vocab, tgt_vocab, opt)
        
    if opt.verbose:
        separator()
        print('Attribution matrix')
        try:
            with np.printoptions(precision=3, suppress=True):
                print(scores)
        except:
            print(scores)
        print()
    
    separator()
    print('Output:',' '.join(output_seq[:]))
    
    plot_attributions(opt, scores, src_seq, output_seq)
    

if __name__=="__main__":
    opt = parse_args()
    if opt.verbose:
        separator()
    main(opt)
    separator()
    