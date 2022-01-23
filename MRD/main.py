import argparse
import numpy as np
from lightgbm import train
import pandas as pd
from data import read_csv
from ctgan import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sdv.metrics.tabular import SVCDetection
from evaluation import Evaluator


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')

    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--data', default='/Users/paubellonchmarchan/Desktop/program_tfg/data_testing/churn-1.csv',help='Path to training data',)
    parser.add_argument('--out', default='/Users/paubellonchmarchan/Desktop/program_tfg/data_synthetic/synthetic_data.csv',help='Path of the output file',)
    
    return parser.parse_args()

def main():
    args = _parse_args()
    data  = read_csv(args.data)
    discrete_columns = []
    atributos=list(data)
    num_atributos=list(data._get_numeric_data())
    categorical_columns=set(atributos).difference(set(num_atributos))
    print(data.describe)
    generator_dim = [int(x) for x in args.generator_dim.split(',')]
    discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
    model = CTGANSynthesizer(
        embedding_dim=args.embedding_dim, generator_dim=generator_dim,
        discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
        generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
        discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
        epochs=args.epochs)
    
    tarin, test=train_test_split(data,test_size=0.9)
    model.fit(tarin, categorical_columns)
    
    if args.save is not None:
        model.save(args.save)

    num_samples = args.num_samples or len(test)
    train_synth = model.sample(num_samples)
    ml_eval=SVCDetection.compute(data, train_synth)
    print(train_synth)
    print("El valor de detección ML: ", ml_eval)
    train_synth.to_csv(args.out, index=False)

    rows_train=len(train_synth.index)
    rows_test=len(test.index)
    test_train=pd.concat([train_synth, test], axis=0)
    test_train.reset_index(drop=True, inplace=True)
    evaluar_data=test_train.head(rows_test)
    evaluar_train_synth=test_train.tail(rows_train)
    
    #Evaluation synthetic and real data
    evaluator=Evaluator(evaluar_data,evaluar_train_synth,cat_cols=categorical_columns)
    evaluator.plot_mean_std()
    evaluator.plot_distributions()
    evaluator.plot_correlation_difference()
    evaluator.plot_cumsums()
   
if __name__ == "__main__":
    main()