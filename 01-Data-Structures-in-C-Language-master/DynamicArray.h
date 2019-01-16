
int Size=10;
int End =-1;

int *Array = new int[Size];

void DoublingArray(){
    Size *= 2;
    int* newArray = new int[Size];
    memcpy(newArray,Array,Size*sizeof(int));
    delete [] Array;
    Array = newArray;
}
void Add(int data){
    ++End;
    if (End+1 >= Size)
        DoublingArray();
    Array[End] = data;

}

void Delete(){
    End--;
}

int Index(int n){
    return Array[n];
}

void Print(){
    for(int i = 0; i <= End ; i++)
        printf("%d ",Array[i]);
}
int *Arr(){
return Array;
}

void ReversePrint(){

for(int i = End ; i >= 0 ; i--)
        printf("%d ",Array[i]);
}
