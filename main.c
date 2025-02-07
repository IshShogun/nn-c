#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

typedef struct {
    float *entries;
    int rank;
    int* shape;
} Tensor;

typedef struct {
    Tensor* image_batches;
    Tensor* label_batches;
} Data;

typedef struct {
    Tensor weights;
    Tensor bias;
} Projection;

typedef struct {
    Projection *projections;
    int n_layers;
} Layers;

const int image_id = 2051;
const int label_id = 2049;

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

                int *shape_ptr = malloc(3*sizeof(int));
                memcpy(shape_ptr, (int[]){batch_size, n_rows * n_cols, 1}, 3*sizeof(int));

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
                            shape_ptr = malloc(3*sizeof(int));
                            memcpy(shape_ptr, (int[]){n_imgs_in_batch, n_rows * n_cols, 1}, 3*sizeof(int));
                        }

                        data->image_batches[batch_id].entries = malloc(n_imgs_in_batch * n_rows * n_cols * sizeof(float));
                        data->image_batches[batch_id].rank = 3;
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

void initaliseRandomProjections(Layers *layers, int seed){
    srand(seed);
    for(int l = 0; l < layers->n_layers; l++){
        Projection *p = &layers->projections[l];
        
        size_t n_elems_weights = 1;
        size_t n_elems_bias = 1;

        int max_rank = (p->weights.rank > p->bias.rank) ? p->weights.rank : p->bias.rank;

        for(int r = 0; r < max_rank; r++){
            if(r < p->weights.rank){
                n_elems_weights *= p->weights.shape[r]; 
            }

            if(r < p->bias.rank){
                n_elems_bias *= p->bias.shape[r]; 
            }

        }

        p->weights.entries = malloc(n_elems_weights * sizeof(float));
        p->bias.entries = malloc(n_elems_bias * sizeof(float));

        int max_elems = (n_elems_weights > n_elems_bias) ? n_elems_weights : n_elems_bias;
        for(int i = 0; i < max_elems; i++){
            if(i < n_elems_bias){
                p->bias.entries[i] = ((float)rand()/(float)(RAND_MAX/2)) - 1.0;   
            }

            if(i < n_elems_weights){
                p->weights.entries[i] = ((float)rand()/(float)(RAND_MAX/2)) - 1.0;   
            }
        }
    }
}



float relu(float z_i){
    return (z_i <= 0.0f) ? 0.0f : z_i;
}

Tensor forward_pass(Layers layers, Tensor batch){
    int batch_size = batch.shape[0];

    Tensor input_tensor;

    input_tensor.rank = batch.rank - 1;
    //we allocate to it first so that its safe to allocate or it could write to a dangerous location
    input_tensor.shape = malloc(input_tensor.rank * sizeof(int));

    //.shape is an *int so no need for &
    memcpy(input_tensor.shape, &batch.shape[1], (input_tensor.rank) * sizeof(int));

    int n_elems = 1;
    for(int i = 0; i < input_tensor.rank; i++){
        n_elems *= input_tensor.shape[i];
    }

    input_tensor.entries = malloc(n_elems * sizeof(float));

    for(int idx = 0; idx < batch_size; idx++){
        memcpy(input_tensor.entries, &batch.entries[idx * n_elems], n_elems * sizeof(float));
        
        for(int layer_i = 0; layer_i < layers.n_layers; layer_i++){
            
        }
    }

    return input_tensor;
}

//we batch the input data and save the change in weights, for after we have run the whole batch we update with the average change in weights
//batch size as a multiple of 32 due to hardware optimisations with the gpu later (warps) -> 64

//image data is an array of 3D tensors (n_batches, batch_size, n_rows*n_cols, 1)
//label data is an array of @D tensors (n_batches, batch_size, 1)
void train(Layers *layers, Data data, size_t n_epochs){
    int n_batches = data.image_batches->shape[0];
    if(n_batches != data.label_batches->shape[0]){
        printf("Number of batches in training and label data do not match\n");
        exit(1);
    }
    for(int epoch_i = 0; epoch_i < n_epochs; epoch_i++){
        for(int batch_i = 0; batch_i < n_batches; batch_i++){
            //batch size in each 
            Tensor img_batch = data.image_batches[batch_i];
            Tensor label_batch = data.label_batches[batch_i];

            if(img_batch.shape[0] != label_batch.shape[0]){
                printf("batch size of in training and label data do not match\n");
                exit(1);
            }

            forward_pass(*layers, img_batch);
        } 
    }
}




int main(void){
    Data training_data = {
        .image_batches=NULL,
        .label_batches=NULL,
    };

    int batch_size = 32;
    read_data(&training_data, "data/training/", batch_size);

    /*batch_id: 40, entry_id: 22530, img_id: 1308, entry: 0.992157*/
    /*printf("Batch 40, Entry 22530: %f\n", training_data.image_batches[40].entries[22530]);*/
    /*printf("n_batches: %i\n", training_data.n_batches);*/

    int next_dim_after_proj[3] = {1024, 512, 10};

    Layers layers = {
       .n_layers=3,
       .projections = malloc(3*sizeof(Projection))
    };

    int prev_dim = 784; //the dimensions of n_rows * n_cols could easily return this in read_data but cba

    //the dimension after the next projection so index 0 is the dimension after proj 0
    for(int i = 0; i < layers.n_layers; i++){
        int next_dim = next_dim_after_proj[i];

        int* p_w_shape = malloc(2 * sizeof(int));
        memcpy(p_w_shape, (int[]){next_dim, prev_dim}, 2 * sizeof(int));

        int* p_b_shape = malloc(sizeof(int));
        memcpy(p_b_shape, (int[]){next_dim}, sizeof(int));

        layers.projections[i] = (Projection){
                .weights=(Tensor){
                    .entries=NULL,
                    .shape=p_w_shape,
                    .rank=2
                },
                .bias=(Tensor){
                    .entries=NULL,
                    .shape=p_b_shape,
                    .rank=1
                },
        };

        prev_dim = next_dim;
    }

    int seed = 42; //rng seed
    initaliseRandomProjections(&layers, seed);

    train(&layers, training_data, 1);
    //weights are initalised so now i need to forward pass 
    //input data matmul with project then activation till final layer 
    return 0;
}

