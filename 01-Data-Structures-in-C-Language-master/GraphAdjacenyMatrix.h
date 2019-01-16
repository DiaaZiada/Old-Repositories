const int Size = 101 ;

int Matrix[Size][Size];

void AddRelation(int mem1,int mem2){
    if(mem1 >= Size || mem2 >= Size)
        return ;
    Matrix[mem1-1][mem2-1] = 1;
}

void RemoveRelation(int mem1,int mem2){
    Matrix[mem1-1][mem2-1] = 0;
}

void ViewData(){
    for(int i = 0; i< Size; i++)
        for(int j = 0; j < Size; j++)
            if(Matrix[i][j] == 1)
                printf("%d has relation with %d\n",i+1,j+1);
}

void Details(){

}

bool Search(int mem1, int mem2){
    return (Matrix[mem1-1][mem2-1] == 1);
}
