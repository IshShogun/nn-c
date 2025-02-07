#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

typedef struct {
    unsigned char *pixels; 
    int img_width;
    int img_height;
    int n_img;
} Images;

typedef struct {
    unsigned char *labels;
    int n_labels;
} Labels;

typedef struct {
    float *weights;
    float *bias;
    int n_rows;
    int n_cols;
} Projection;

const int image_id = 2051;
const int label_id = 2049;

int read_header_bytes(FILE *file){
    unsigned char b[4]; 
    fread(b, sizeof(char), 4, file);
    
    //Loads b[0] from register shifts it 24 bits at the address &data to the left and so on
    return (int)(b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]);
}

void read_data(Images *image_data, Labels *labels, const char *data_path){
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

                
                //just have all the image data in one array i can seperate it as i like
                if(image_data->pixels == NULL){
                    size_t n_pixels = n_images * n_rows * n_cols;

                    image_data->pixels = malloc(n_pixels * sizeof(unsigned char));

                    image_data->n_img = n_images;
                    image_data->img_height = n_rows;
                    image_data->img_width = n_cols;
                } else {
                    int total_imgs = n_images + image_data->n_img;
                    size_t n_pixels = total_imgs * n_images * n_cols; 

                    image_data->pixels = realloc(image_data->pixels, n_pixels * sizeof(unsigned char));

                    image_data->n_img = total_imgs;
                }

                int counter = 0;
                unsigned char buffer;

                while(fread(&buffer, sizeof(char), 1, file) == 1){
                    image_data->pixels[counter] = buffer; 
                    counter++;
                }
            } else if(magic_number == label_id) {
                int n_labels = read_header_bytes(file);
                
                if(labels->labels == NULL){
                    labels->labels = malloc(n_labels * sizeof(unsigned char));
                    labels->n_labels = n_labels;
                } else {
                    int total_labels = n_labels + labels->n_labels;

                    labels->labels = realloc(labels->labels, total_labels * sizeof(unsigned char));
                    labels->n_labels = total_labels;
                }

                int counter = 0;
                unsigned char buffer;

                while(fread(&buffer, sizeof(char), 1, file) == 1){
                    labels->labels[counter] = buffer;
                    counter++;
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

void initaliseRandomProjections(Projection *projections, int seed, int n_layers){
    srand(seed);
    for(int l = 0; l < n_layers; l++){
        Projection *p = &projections[l];
        p->weights = malloc(p->n_rows * p->n_cols * sizeof(float));
        p->bias = malloc(p->n_rows * sizeof(float));
        for(int i = 0; i < p->n_rows; i++){
            for(int j = 0; j < p->n_cols; j++){
                //same as a * (N/RAND_MAX) - 1. N is in [0, RAND_MAX] so N/... is between 0 and 1 and we multiply by the scale
                //substract 1 so we in [-1, 1]
                float w_ij = ((float)rand()/(float)(RAND_MAX/2)) - 1.0;
                //it fails here
                p->weights[i * p->n_cols + j] = w_ij;   
            }
            p->bias[i] =  ((float)rand()/(float)(RAND_MAX/2)) - 1.0;
        }
    }
}

float relu(float z_i){
    return (z_i <= 0.0f) ? 0.0f : z_i;
}


int main(void){
    Images image_data = {0};
    Labels labels = {0};
    read_data(&image_data, &labels, "data/training/");


    int n_layers = 3;
    int next_dim_after_proj[3] = {1024, 512, 10};

    Projection *projections = malloc(n_layers*sizeof(Projection));

    //the dimension after the next projection so index 0 is the dimension after proj 0
    for(int i = 0; i < n_layers; i++){
        int n_cols = (i == 0) ? image_data.img_width*image_data.img_height : projections[i - 1].n_rows;
        int n_rows = next_dim_after_proj[i];
        projections[i] = (Projection){
                .weights=NULL,
                .bias=NULL,
                .n_cols=n_cols,
                .n_rows=n_rows
        };
    }

    int seed = 42; //rng seed
    initaliseRandomProjections(projections, seed, n_layers);

    //weights are initalised so now i need to forward pass 
    //input data matmul with project then activation till final layer 
    return 0;
}

