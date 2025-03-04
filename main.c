#include <stdio.h>
#include <math.h> // For fabsf()
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define EPSILON 1e-6

typedef struct {
    float *entries;
    int rank;
    int* shape;
} Tensor;

typedef struct {
    Tensor* image_batches;
    Tensor* label_batches;
    size_t n_batches;
} Data;

typedef struct {
    Tensor *weights;
    Tensor *bias;
    //arr of pointers
    Tensor **a_batch;
    Tensor **z_batch;
} Layer;

typedef struct {
    Layer *layers;
    int n_layers;
} Layers;

const int image_id = 2051;
const int label_id = 2049;

typedef Tensor* (*activaton_t)(Tensor* z_i);

int read_header_bytes(FILE *file){
    unsigned char b[4]; 
    fread(b, sizeof(char), 4, file);
    
    //Loads b[0] from register shifts it 24 bits at the address &data to the left and so on
    return (int)(b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]);
}

void read_data(Data *data, const char *data_path, int batch_size){
    //this a stream structure when you call readdir(d) it returns a pointer to the next dirent -> data coudl be all over
    //a dirent has the information for a file or directory it has fields (char d_name[], d_ino[], d_type) -> d_type says if its a file or dir d_name is the name
    DIR *d;

    struct dirent *dir;


    //this makes a syscall to the kernel to get the directory info here
    d = opendir(data_path);

    if(d){
        while ((dir = readdir(d)) != NULL){
            if(dir->d_type != DT_REG){
                continue;
            }

            FILE *file;
            {
                char path[1024]; 
                snprintf(path, sizeof(path), "%s/%s", data_path, dir->d_name);


                file = fopen(path, "r");
                if(file == NULL){
                    printf("Could not open file: %s", path);
                }
            }

            int magic_number = read_header_bytes(file);

            if(magic_number == image_id){
                int n_images = read_header_bytes(file);
                int n_rows = read_header_bytes(file);
                int n_cols = read_header_bytes(file);
                size_t n_batches = n_images / batch_size;
                
                data->n_batches = n_batches;

                //goal tensor (n_batches, batch_size, n_rows, n_cols)
                if(data->image_batches == NULL){
                    //each tensor here is a batch
                    data->image_batches = malloc(n_batches * sizeof(Tensor));
                }


                //there are batch_size imgs in a batch
                int img_id = 0;
                int batch_id = 0;
                int entry_id = 0;

                bool alloc = true;

                int *shape_ptr = malloc(2*sizeof(int));
                memcpy(shape_ptr, (int[]){batch_size, n_rows * n_cols}, 2*sizeof(int));

                unsigned char buffer;
                while(fread(&buffer, sizeof(char), 1, file) == 1){
                    if(img_id % batch_size == 0 && alloc){
                        if(img_id > 0){
                            batch_id++;
                            entry_id = 0;
                        }

                        int n_imgs_in_batch = batch_size;
                        if(n_images - img_id < batch_size){
                            n_imgs_in_batch = n_images - img_id; 
                            shape_ptr = malloc(2*sizeof(int));
                            memcpy(shape_ptr, (int[]){n_imgs_in_batch, n_rows * n_cols}, 2*sizeof(int));
                        }

                        data->image_batches[batch_id].entries = malloc(n_imgs_in_batch * n_rows * n_cols * sizeof(float));
                        data->image_batches[batch_id].rank = 2;
                        data->image_batches[batch_id].shape = shape_ptr;
                        alloc = false;
                    } 


                    data->image_batches[batch_id].entries[entry_id] = (float)buffer/255; 
                    entry_id++;

                    if(entry_id % (n_rows * n_cols) == 0 && entry_id != 0){
                        img_id++;
                        alloc = true;
                    }
                }
            } else if(magic_number == label_id) {
                int n_labels = read_header_bytes(file);
                size_t n_batches = n_labels / batch_size;
                data->n_batches = n_batches;
                
                //goal tensor is (n_batches, batch_size, 1)
                if(data->label_batches== NULL){
                    data->label_batches = malloc(n_batches * sizeof(Tensor));
                } 


                int *shape_ptr = malloc(2*sizeof(int));
                memcpy(shape_ptr, (int[]){batch_size, 1}, 2*sizeof(int));

                int label_id = 0;
                int batch_id = 0;
                int entry_id = 0;

                unsigned char buffer;

                while(fread(&buffer, sizeof(char), 1, file) == 1){
                    if(label_id % batch_size == 0){
                        if(label_id > 0){
                            entry_id = 0;
                            batch_id++;
                        }


                        int n_labels_in_batch = batch_size;
                        if(n_labels - label_id < batch_size){
                            n_labels_in_batch = n_labels - label_id; 
                            //old shape_ptr still referenceed in tensor i can free there
                            shape_ptr = malloc(2*sizeof(int));
                            memcpy(shape_ptr, (int[]){n_labels_in_batch, 1}, 2*sizeof(int));
                        }

                        data->label_batches[batch_id].entries = malloc(batch_size * 1 * sizeof(float));
                        data->label_batches[batch_id].rank = 2;
                        data->label_batches[batch_id].shape = shape_ptr;
                    }
                    
                    data->label_batches[batch_id].entries[entry_id] = buffer;
                    label_id++;
                    entry_id++;
                }
            } else {
                printf("Unrecognised file type\n");
                exit(1);
            }

            fclose(file);
        }
    }

    closedir(d);
}

void initalise_random_layers(Layers *layers, int seed){
    srand(seed);
    for(int l = 0; l < layers->n_layers; l++){
        Layer *l_i = &layers->layers[l];
        
        size_t n_elems_weights = 1;
        size_t n_elems_bias = 1;

        int max_rank = (l_i->weights->rank > l_i->bias->rank) ? l_i->weights->rank : l_i->bias->rank;

        for(int r = 0; r < max_rank; r++){
            if(r < l_i->weights->rank){
                n_elems_weights *= l_i->weights->shape[r]; 
            }

            if(r < l_i->bias->rank){
                n_elems_bias *= l_i->bias->shape[r]; 
            }

        }

        l_i->weights->entries = malloc(n_elems_weights * sizeof(float));
        l_i->bias->entries = malloc(n_elems_bias * sizeof(float));

        int max_elems = (n_elems_weights > n_elems_bias) ? n_elems_weights : n_elems_bias;
        for(int i = 0; i < max_elems; i++){
            if(i < n_elems_bias){
                l_i->bias->entries[i] = ((float)rand()/(float)(RAND_MAX/2)) - 1.0;   
            }

            if(i < n_elems_weights){
                l_i->weights->entries[i] = ((float)rand()/(float)(RAND_MAX/2)) - 1.0;   
            }
        }
    }
}


Tensor* matmul(Tensor A, Tensor B){
    //In AB we go down the rows of matrix A and across the columns of B
    if (A.rank != 2 || B.rank != 1) {
        printf("A must be rank 2, is %i\n B must be rank, is %i\n", A.rank, B.rank);
        exit(1);
    }

    int a_height = A.shape[0]; 
    int a_width = A.shape[1]; 

    int b_height = B.shape[0]; 
    if(a_width != b_height){
        printf("The number of columns of A must match the rows of B, A shape: (%i, %i), B rows: (%i, %i)\n", A.shape[0], A.shape[1], B.shape[0], B.shape[1]);
        exit(1);
    }



    Tensor *C = malloc(sizeof(Tensor));
    C->shape = malloc(sizeof(int));
    memcpy(C->shape, &A.shape[0], sizeof(int));
    
    C->rank = 1;
    C->entries = malloc(C->shape[0] * sizeof(float));

    //choose the row of A
    for(int i = 0; i < a_height; i++){
        float c_ij = 0;
        //choose row of B, n cols is 1 so we collapse the extra loop
        for(int j = 0; j < b_height; j++){
            float a_ik = A.entries[i * a_width + j];
            float b_kj = B.entries[j];

            c_ij += a_ik * b_kj;
        }

        C->entries[i] = c_ij;
    }

    return C;
}

Tensor* relu(Tensor *z_i){
    if(z_i->rank != 1){
        printf("Error z must be rank 1, is rank %i\n", z_i->rank);
        exit(1);
    }

    int n_elems = z_i->shape[0];

    Tensor *a = malloc(sizeof(Tensor));
    a->entries = malloc(n_elems * sizeof(float));
    a->shape = malloc(sizeof(int));
    memcpy(a->shape, &z_i->shape[0], sizeof(int));
    a->rank = 1; 
    for(int i = 0; i < n_elems; i++){
        a->entries[i] = (z_i->entries[i] > 0) ? z_i->entries[i] : 0;    
    }

    return a;
}

float change_in_loss_for_prediction(Tensor *predictions, Tensor batched_true_labels, int row, int output_dim, int batch_size, int batch_idx){
    float change_in_loss_for_ai = 0.0;
    for(int k = 0; k < output_dim; k++){
        float d;
        if(row == k){
            d = 1;
        } else {
            d = 0;
        }

        //yk is the true label 
        int idx = batch_idx * batch_size + k;
        //true_labels_stores true label, k is the index of the prediction
        float yk = ((int)(batched_true_labels.entries[idx] == k)) ? 1 : 0;
        change_in_loss_for_ai +=  yk * (d * predictions->entries[row]);
    }

    return change_in_loss_for_ai *= -1;

}

//label batch dim (batch_size, 10)
//cross entropy loss
float get_change_in_loss_for_weight(Layers *layers, Tensor batched_true_labels, int batch_idx, int depth, int weight_index){
    float delta = 0.0;

    int batch_size = batched_true_labels.shape[0];
    int output_dim = batched_true_labels.shape[1];
    /*printf("label batch dim (%i, %i)\n", batch_size, output_dim);*/
    /*printf("label rank %i\n", batched_true_labels.rank);*/
    int n_layers = layers->n_layers;
    int w_row = weight_index / layers->layers[n_layers - (depth + 1)].weights->shape[0];
    int w_col = weight_index / layers->layers[n_layers - (depth + 1 )].weights->shape[1];

    Tensor *predictions = layers->layers[n_layers - 1].a_batch[batch_idx];
    int curr_layer = n_layers - 1;
    switch (depth){
        case 0: {
            //divide by the raw index weight by the n rows in the layer weights to get the row index i
            float change_in_loss_for_ai = change_in_loss_for_prediction(predictions, batched_true_labels, w_row, output_dim, batch_size, batch_idx);

            float zi = (layers->layers[curr_layer].z_batch[batch_idx]->entries[w_row]);
            float change_in_ai_for_zi = (zi > 0.0) ? 1.0 : 0.0;
             

            curr_layer--;
            //same as aj in penultimate layer
            float change_in_zi_for_weight = (layers->layers[curr_layer].a_batch[batch_idx]->entries[w_col]);
            
            delta = change_in_loss_for_ai * change_in_ai_for_zi * change_in_zi_for_weight;
            break;
        }
        case 1: {
            Tensor *z = layers->layers[curr_layer].z_batch[batch_idx];
            
            float change_in_loss_for_ai = 0.0;
            int final_layer_weights_width = layers->layers[curr_layer].weights->shape[0];
            //rows of z
            for(int j = 0; j < z->shape[0]; j++){
                //change in zj for ai is just the layer weight w_ji. index i here is just the row inde of the weight we are minimising the 
                //loss for
                float change_in_zj_for_ai = layers->layers[curr_layer].weights->entries[j * final_layer_weights_width + w_row];

                float zj = (layers->layers[curr_layer].z_batch[batch_idx]->entries[j]);
                float change_in_aj_for_zj = (zj > 0.0) ? 1.0 : 0.0;

                float change_in_loss_for_aj = change_in_loss_for_prediction(predictions, batched_true_labels, j, output_dim, batch_size, batch_idx);

                change_in_loss_for_ai += (change_in_zj_for_ai * change_in_aj_for_zj * change_in_loss_for_aj);
            }

            curr_layer--;
            float zi = (layers->layers[curr_layer].z_batch[batch_idx]->entries[w_row]);
            //hold on may be some floating point shenanigans here
            float change_in_ai_for_zi = (zi > 0.0) ? 1.0 : 0.0; 

            float change_in_zi_for_weight = (layers->layers[curr_layer].a_batch[batch_idx]->entries[w_col]);
            delta += change_in_zi_for_weight * change_in_ai_for_zi *  change_in_loss_for_ai;
            break;
           }
        case 2: {
            float change_in_loss_for_a = 0.0;

            for(int j = 0; j < predictions->shape[0]; j++){
                change_in_loss_for_a += change_in_loss_for_prediction(predictions, batched_true_labels, j, output_dim, batch_size, batch_idx);
            }

            {
                Tensor *z = layers->layers[curr_layer].z_batch[batch_idx];
                Tensor *a = layers->layers[curr_layer - 1].a_batch[batch_idx];

                float change_in_a_for_z_times_change_in_a_for_z = 0.0;
                for(int i = 0; i < z->shape[0]; i++){
                    for(int j = 0; j < a->shape[0]; j++){
                        float zi = layers->layers[curr_layer].z_batch[batch_idx]->entries[i];
                        Tensor *layer_weights = layers->layers[curr_layer].weights;
                        int layer_width = layer_weights->shape[0];

                        float change_in_ai_for_zi = ( zi > 0.0 ? 1.0 : 0.0);
                        //just weight wij for this layer
                        float change_in_zi_for_aj = layer_weights->entries[layer_width * i + j];

                        change_in_a_for_z_times_change_in_a_for_z += change_in_zi_for_aj * change_in_ai_for_zi;
                    }
                }
                delta = change_in_a_for_z_times_change_in_a_for_z;
            }

            curr_layer--;
            {
                float change_in_z_for_ai_times_change_in_a_for_z = 0.0;
                Tensor *z = layers->layers[curr_layer].z_batch[batch_idx];
                Tensor *a = layers->layers[curr_layer - 1].a_batch[batch_idx];

                for(int j = 0; j < z->shape[0]; j++){
                    float zj = layers->layers[curr_layer].z_batch[batch_idx]->entries[j];
                    Tensor *layer_weights = layers->layers[curr_layer].weights;
                    int layer_width = layer_weights->shape[0];


                    float change_in_aj_for_zj = zj > 0.0 ? 1.0 : 0.0;
                    float change_in_zj_for_ai = layer_weights->entries[layer_width * w_row + j];
                    change_in_z_for_ai_times_change_in_a_for_z += change_in_zj_for_ai * change_in_aj_for_zj;
                }

                delta += change_in_z_for_ai_times_change_in_a_for_z;
            }

            curr_layer--;
            
            {
                float zi = layers->layers[curr_layer].z_batch[batch_idx]->entries[w_row];
                float aj = layers->layers[curr_layer].a_batch[batch_idx]->entries[w_row];

                float change_in_ai_for_zi = (zi > 0.0 ? 1.0 : 0.0);
                delta *= change_in_ai_for_zi;

                float change_in_zi_for_weight = aj;
                delta *= change_in_ai_for_zi;

            }
            break;
        }
        default:
            printf("cannot support depth/layers over 3\n");
            exit(1);
    }

    return delta;
}

//we use cross entropy as our loss
void backpropogation(Layers *layers, Tensor label_batch, activaton_t activation, float learning_rate){
    int batch_size = label_batch.shape[0];
    int final_vec_dim = label_batch.shape[1];
    int n_layers = layers->n_layers;

    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        /*printf("\n\ncomputing batch %i/%i\n\n", batch_idx, batch_size);*/
        float* updated_weights[n_layers];
        for(int i = 0; i < n_layers; i++){
            Layer curr_layer = layers->layers[i];
            if(curr_layer.weights->rank != 2){
                printf("Layer %i weights must be rank 2", i);
                exit(1);
            }

            int n_entries = curr_layer.weights->shape[0] * curr_layer.weights->shape[1];
            updated_weights[i]= malloc(n_entries*sizeof(float));
            memcpy(updated_weights[i], curr_layer.weights->entries, n_entries * sizeof(float));
        }

        //back to front
        for(int depth = 0; depth < n_layers; depth++){
            Layer curr_layer = layers->layers[n_layers - (depth + 1)];
            int n_entries = curr_layer.weights->shape[0] * curr_layer.weights->shape[1];

            for(int weight_index = 0; weight_index < n_entries; weight_index++){
                float change_in_loss_for_weight = get_change_in_loss_for_weight(layers, label_batch, batch_idx, depth, weight_index);
                //same as averaging later
                updated_weights[n_layers - (depth + 1)][weight_index] += (learning_rate * change_in_loss_for_weight) / batch_size; 
            }
        }

        // Free old weights and assign updated weights
        for (int i = 0; i < n_layers; i++) {
            int n_entries = layers->layers[i].weights->shape[0] * layers->layers[i].weights->shape[1];
            memcpy(layers->layers[i].weights->entries, updated_weights[i], n_entries * sizeof(float));
            free(updated_weights[i]);
        }

    }
}

//input is (batch_size, 784, 1)
void forward_pass(Layers layers, Tensor batch){
    int batch_size = batch.shape[0];

    int output_dim;
    {
        Tensor *final_proj_w = layers.layers[layers.n_layers - 1].weights;
        int final_proj_rank = final_proj_w->rank;
        output_dim = final_proj_w->shape[final_proj_rank - 1];
    }


    int input_n_elems = 1;
    for(int i = 1; i < batch.rank - 1; i++){
        input_n_elems *= batch.shape[i];
    }

    for(int idx = 0; idx < batch_size; idx++){
        Tensor *input_vector = malloc(sizeof(Tensor));
        input_vector->shape = malloc(sizeof(int));

        memcpy(input_vector->shape, &batch.shape[1], sizeof(int));

        input_vector->rank = 1;
        input_vector->entries = malloc(input_n_elems * sizeof(float));

        memcpy(input_vector->entries, &batch.entries[input_n_elems * idx], input_n_elems * sizeof(float));

        for(int i = 0; i < layers.n_layers; i++){
            Layer layer_i = layers.layers[i];  
            Tensor *z = matmul(*layer_i.weights, *input_vector);

            layer_i.z_batch[idx] = z;

            //a fresh new activation vector here
            Tensor *a = relu(z); 
            layer_i.a_batch[idx] = a;

            input_vector = a;
        }

    }
}

Tensor* predict(Layers layers, Tensor *input){
    for(int i = 0; i < layers.n_layers; i++){
        Layer layer_i = layers.layers[i];  
        Tensor *z = matmul(*layer_i.weights, *input);
        free(input);

        //a fresh new activation vector here
        Tensor *a = relu(z); 
        input = a;
        free(z);
    }

    return input;
}

//we batch the input data and save the change in weights, for after we have run the whole batch we update with the average change in weights
//batch size as a multiple of 32 due to hardware optimisations with the gpu later (warps) -> 64

//image data is an array of 3D tensors (n_batches, batch_size, n_rows*n_cols, 1)
//label data is an array of 3D tensors (n_batches, batch_size, 10)
void train(Layers *layers, Data data, size_t n_epochs, float learning_rate){

    int n_batches = data.n_batches;
    printf("n batches: %i\n", n_batches);

    for(int epoch_i = 0; epoch_i < n_epochs; epoch_i++){
        for(int batch_i = 0; batch_i < n_batches; batch_i++){
            /*printf("Fetching batch %i\n", batch_i);*/
            //batch size in each 
            Tensor img_batch = data.image_batches[batch_i];
            Tensor label_batch = data.label_batches[batch_i];

            size_t batch_size = img_batch.shape[0];
            if(img_batch.shape[0] != label_batch.shape[0]){
                printf("batch size of in training and label data do not match\n");
                exit(1);
            }

            printf("Propogating forward batch %i, epoch %i\n", batch_i, epoch_i);
            forward_pass(*layers, img_batch);

            printf("Propogating backwards batch %i, epoch %i\n", batch_i, epoch_i);
            backpropogation(layers, label_batch, relu, learning_rate);

            for(int k = 0; k < layers->n_layers; k++){
                /*printf("Layer %i\n", k);*/
                Layer *curr_layer = &layers->layers[k];


                //a_batch is an array of pointers
                for(size_t ptr_idx = 0; ptr_idx < batch_size; ptr_idx++){
                    /*printf("print saved projection %li / %li\n", ptr_idx, batch_size);*/
                    free(curr_layer->a_batch[ptr_idx]);
                    free(curr_layer->z_batch[ptr_idx]);
                }
            }
        }

        int correct_classifications = 0;
        int wrong_classifications = 0;
        //get test accuarcy
        for(int batch_i = 0; batch_i < n_batches; batch_i++){
            Tensor img_batch = data.image_batches[batch_i];
            Tensor label_batch = data.label_batches[batch_i];

            size_t batch_size = img_batch.shape[0];
            int input_n_elems = 1;
            for(int i = 1; i < img_batch.rank - 1; i++){
                input_n_elems *= img_batch.shape[i];
            }       

            for(size_t idx = 0; idx < batch_size; idx++){
                Tensor *input_vector = malloc(sizeof(Tensor));
                input_vector->shape = malloc(sizeof(int));

                memcpy(input_vector->shape, &img_batch.shape[1], sizeof(int));

                input_vector->rank = 1;
                input_vector->entries = malloc(input_n_elems * sizeof(float));

                memcpy(input_vector->entries, &img_batch.entries[input_n_elems * idx], input_n_elems * sizeof(float));

                Tensor *predictions = predict(*layers, input_vector);

                float argmax = -1;
                int classification = -1;
                for(int k = 0; predictions->shape[0]; k++){
                    float arg = predictions->entries[k];
                    if(arg > argmax){
                        argmax = arg;
                        classification = k;
                    } 
                }

                int true_classification = label_batch.entries[batch_i * batch_size + idx];
                if(classification == true_classification){
                    correct_classifications += 1;
                } else {
                    wrong_classifications += 1;
                }

            }

        }

        float test_set_accuracy = (float)correct_classifications / ((float)wrong_classifications + (float)correct_classifications);
        printf("Test set accuracy, %f\n", test_set_accuracy);

    }

}



int main(void){
    Data training_data = {
        .image_batches=NULL,
        .label_batches=NULL,
    };

    int batch_size = 8;
    read_data(&training_data, "data/training/", batch_size);

    /*batch_id: 40, entry_id: 22530, img_id: 1308, entry: 0.992157*/
    /*printf("Batch 40, Entry 22530: %f\n", training_data.image_batches[40].entries[22530]);*/
    /*printf("n_batches: %i\n", training_data.n_batches);*/

    int next_dim_after_proj[3] = {1024, 512, 10};

    Layers layers = {
       .n_layers=3,
       .layers= malloc(3*sizeof(Layer))
    };

    int prev_dim = 784; //the dimensions of n_rows * n_cols could easily return this in read_data but cba


    //the dimension after the next projection so index 0 is the dimension after proj 0
    for(int i = 0; i < layers.n_layers; i++){
        int next_dim = next_dim_after_proj[i];

        int* p_w_shape = malloc(2 * sizeof(int));
        memcpy(p_w_shape, (int[]){next_dim, prev_dim}, 2 * sizeof(int));

        int* p_b_shape = malloc(sizeof(int));
        memcpy(p_b_shape, (int[]){next_dim}, sizeof(int));

        layers.layers[i].weights = malloc(sizeof(Tensor));
        layers.layers[i].weights->shape = p_w_shape;
        layers.layers[i].weights->rank = 2;

        layers.layers[i].bias = malloc(sizeof(Tensor));
        layers.layers[i].bias->shape = p_b_shape;
        layers.layers[i].bias->rank = 1;

        layers.layers[i].a_batch = malloc(batch_size * sizeof(Tensor *));
        layers.layers[i].z_batch = malloc(batch_size * sizeof(Tensor *));

        prev_dim = next_dim;
    }

    int seed = 42; //rng seed
    initalise_random_layers(&layers, seed);

    train(&layers, training_data, 10, 0.1);
    //weights are initalised so now i need to forward pass 
    //input data matmul with project then activation till final layer 
    return 0;
}

