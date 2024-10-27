import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse

import matplotlib.pyplot as plt

import os
import time

# Record the start time
start_time = time.time()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
args = parser.parse_args()

# Load primus

primus = CTC_PriMuS(args.corpus,args.set,args.voc, args.semantic, val_split = 0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height,primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5

# Model
inputs, seq_len, targets, decoded, loss, rnn_keep_prob, attention_weights = ctc_model.ctc_crnn(params)
train_opt = tf.compat.v1.train.AdamOptimizer().minimize(loss)

# Restore weights
saver = tf.compat.v1.train.Saver(max_to_keep=None)
sess.run(tf.compat.v1.global_variables_initializer())

# Training loop
for epoch in range(max_epochs):
    batch = primus.nextBatch(params)

    _, loss_value = sess.run([train_opt, loss],
                             feed_dict={
                                inputs: batch['inputs'],
                                seq_len: batch['seq_lengths'],
                                targets: ctc_utils.sparse_tuple_from(batch['targets']),
                                rnn_keep_prob: dropout,
                            })


    print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))

    log_message0 = f'Epoch: {epoch}, Loss: {loss_value}\n'

    # Save to file
    with open('trainingloss_log.txt', 'a') as log_file:
        log_file.write(log_message0)



    if epoch % 1000 == 0: # Validation at every 1000 epochs
        # VALIDATION
        print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))
        print ('Validating...')

        validation_batch, validation_size = primus.getValidation(params)
        
        val_idx = 0
        
        val_ed = 0
        val_len = 0
        val_count = 0
            
        while val_idx < validation_size:
            mini_batch_feed_dict = {
                inputs: validation_batch['inputs'][val_idx:val_idx+params['batch_size']],
                seq_len: validation_batch['seq_lengths'][val_idx:val_idx+params['batch_size']],
                rnn_keep_prob: 1.0            
            }            
                        
            
            prediction, attention_weights_val = sess.run([decoded, attention_weights], feed_dict=mini_batch_feed_dict)

            # Debugging: Print the shape of attention_weights_val
            print("Shape of attention_weights_val:", attention_weights_val.shape)
    
            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    

            for i in range(len(str_predictions)):
                ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
                val_ed = val_ed + ed
                val_len = val_len + len(validation_batch['targets'][val_idx+i])
                val_count = val_count + 1
                
            val_idx = val_idx + params['batch_size']

        elapsed_time = time.time() - start_time
        print ('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')        
        print('Elapsed time: {:.2f} seconds'.format(elapsed_time))
        print ('Saving the model...')


        log_message = (
            f'[Epoch {epoch}] {1. * val_ed / val_count} ({100. * val_ed / val_len} SER) from {val_count} samples.\n'
            f'Elapsed time: {elapsed_time:.2f} seconds\n'
            'Saving the model...\n'
            '------------------------------\n'
        )

        # Save to file
        with open('training_log.txt', 'a') as log_file:
            log_file.write(log_message)


        saver.save(sess,args.save_model,global_step=epoch)
        print ('------------------------------')

        # Debugging: Print the shape of the attention weights
        #print("Shape of attention_weights_val:", attention_weights_val.shape)
        #print("Shape of attention_weights_val[0]:", attention_weights_val[0].shape)


        # Visualize the attention weights for the last mini-batch
        if attention_weights_val.ndim == 2:  # Check if it's a 2D array
            for i in range(attention_weights_val.shape[1]):
                plt.imshow(attention_weights_val[:, i].reshape(-1, 1), cmap='hot', interpolation='nearest', aspect='auto')
                plt.title(f'Attention Weights for Sample {i} at Epoch {epoch}')
                plt.xlabel('Time Steps')
                plt.ylabel('Attention Score')
                plt.colorbar()
                plt.savefig(f'attention_weights_epoch_{epoch}_sample_{i}.png')
                plt.clf()
        elif attention_weights_val.ndim == 1:  # Handle 1D case (flattened)
            plt.imshow(attention_weights_val.reshape(-1, 1), cmap='hot', interpolation='nearest', aspect='auto')
            plt.title(f'Attention Weights at Epoch {epoch}')
            plt.xlabel('Time Steps')
            plt.ylabel('Attention Score')
            plt.colorbar()
            plt.savefig(f'attention_weights_epoch_{epoch}.png')
            plt.show()
        else:
            print(f"Unexpected shape for attention weights: {attention_weights_val.shape}")

        # record end time

        end_time = time.time()
        print(f"Time taken for epoch {epoch}: {end_time - start_time} seconds")


