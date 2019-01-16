struct Node{
    int data;
    struct Node *next;
}*List[101];
const int Size = 101;


struct Node *CreatingNewNode(int data){
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode -> data = data;
    newNode -> next = NULL;
    return newNode;
}


struct Node* Insert(struct Node* node, int key)
{
    if (node == NULL){
        return CreatingNewNode(key);;
    }
    Insert(node->next,key);
    return node;
}

void AddRelation(int mem1, int mem2){
    List[mem1-1] = Insert(List[mem1-1],mem2);
}

void RemoveRelation(int mem1,int mem2){
    struct Node *newNode = List[mem1-1];
    if (mem1 == 1){
        List[mem1-1] = newNode -> next;
        free(newNode);
        return;
   }
    for(int i =0; i < mem2-2; i++)
        newNode =  newNode -> next;
    struct Node* newNode2 = newNode -> next;
    newNode -> next = newNode2 -> next;
    free(newNode2);
}

void ViewData(){
    struct Node *newNode;
    for(int i = 0; i<Size; i++){
        newNode = List[i];
        while(newNode != NULL){
            printf("%d has relation with %d\n",i+1,newNode -> data);
            newNode = newNode -> next;
        }
    }
}
void Details(){

}
bool Search(int mem1,int mem2){
    struct Node *newNode;

    newNode = List[mem1-1];
    while(newNode != NULL){
        if(newNode -> data == mem2)
            return true;
        newNode = newNode -> next;
    }
    return false;
}
