struct Node{
    int data;
    struct Node *next;
};

struct Node *Front = NULL;
struct Node *Rear = NULL;

bool isEmpty(){
    return (Front == NULL && Rear == NULL);
}

void Enqueue(int data){
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));

    newNode -> data = data;
    newNode -> next = NULL;
    if(isEmpty()){
        Front = Rear = newNode;
        return;
    }
    Rear -> next = newNode;
    Rear = newNode;
}

void Dequeue(){
    if(Front == NULL)
        return;
    if(Front == Rear)
        Front = Rear = NULL;
    else{
        struct Node *newNode = Front;
        Front = Front -> next;
        free(newNode);
    }
}


int FrontIndex(){
    return Front -> data;
}


void Print(){
    struct Node *newNode = Front;

    while(newNode != NULL){
        printf("%d ",newNode -> data);
        newNode = newNode -> next;
   }
   printf("\n");
}

bool Search(int data){
    struct Node *newNode = Front;

    while(newNode != NULL){
        if(newNode -> data == data)
            return true;
        newNode = newNode -> next;
   }
   return false;
}

void Details(){


}
