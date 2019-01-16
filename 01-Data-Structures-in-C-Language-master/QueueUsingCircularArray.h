int Size = 10;
int *Array = new int[Size];
int Front = -1;
int Rear = -1;


void DoublingArray(){
    Size *= 2;
    cout<<Size<<endl;
    int* newArray = new int[Size];
    memcpy(newArray,Array,Size*sizeof(int));
    delete [] Array;
    Array = newArray;

}

bool isEmpty(){
    return (Front == -1 && Rear==-1);
}

bool isFull()
{
    return ((Rear+1)%Size == Front);
}

void Enqueue(int data){

    if (isFull())
        DoublingArray();
    else if (isEmpty())
        Front = Rear = 0;
    else
        Rear = (Rear + 1) % Size;

    Array[Rear] = data;
}

void Dequeue(){

    if(isEmpty())
        return;
    else if(Front == Rear)
        Front = Rear =-1;
    else
        Front += 1;
}


int FrontIndex(){

		if(Front == -1){
			cout<<"Error: cannot return front from empty queue\n";
			return -1;
		}
    return Array[Front];
}


void Print(){
    int counter = ((Rear + Size - Front) % Size) + 1;
    for (int i = 0; i< counter; i++)
        printf("%d ",Array[((Front + i) % Size)]);
    printf("\n");
}
bool Search(int data){

    int counter = ((Rear + Size - Front) % Size) + 1;
    for (int i = 0; i< counter; i++)
        if(Array[((Front + i) % Size)] == data)
            return true;
    return false;
}

void Details(){


}
