struct Node{
    int data;
    struct Node *next;
    struct Node *prev;
};

struct Node *head = NULL;

struct Node *CreatingNewNode(int data){

    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode -> data = data;
    newNode -> next = NULL;
    newNode -> prev = NULL;
    return newNode;
}

void InsertAtHead(int data){
    struct Node *newNode = CreatingNewNode(data);
    if(head == NULL){
     head = newNode;
     return;
    }
    head -> prev = newNode;
    newNode ->next = head;
    head = newNode;
    return;
}

void InsertAtTail(int data){
    struct Node * newNode1 = CreatingNewNode(data);
    if(head == NULL){
     head = newNode1;
     return;
    }
    struct Node *newNode2 = head;
    while (1){
        if(newNode2->next != NULL)
            newNode2 = newNode2 ->next;
        else
            break;
    }
     newNode1 -> next = newNode2 -> next;
     newNode2 -> next = newNode1;
     newNode1 -> prev = newNode2;

}

int Index(int n){
    struct Node *newNode = head;
    for(int i =0; i < n-1; i++)
        newNode =  newNode -> next;
    return newNode -> data;
}

void InsertAtIndex(int data, int n){
    struct Node *newNode = CreatingNewNode(data);
    newNode -> data = data;
    if(n == 1){
        head -> prev = newNode;
        newNode ->next = head;
        head = newNode;
        return;
   }
    struct Node *newNode2 = head;

    for(int i = 0; i<n-2; i++)
        newNode2 = newNode2 -> next;

    struct Node *newNode3 = newNode2 ->next;
    newNode3 -> prev = newNode;
    newNode -> next = newNode2 -> next;
    newNode2 -> next = newNode;
    newNode -> prev = newNode2;
}

void Print(){
    struct Node *newNode = head;
    while (newNode != NULL){
        cout << newNode->data <<" ";
        newNode = newNode -> next;
    }
    cout << endl;
}

void Delete(int n){
    struct Node *newNode = head;
    struct Node *newNode2;
    if (n == 1){
        head = newNode -> next;
        newNode2 = newNode -> next;
        head = newNode2;
        newNode2 -> prev = head;
        free(newNode);
        return;
   }
    for(int i =0; i < n-2; i++)
        newNode =  newNode -> next;
    struct Node* newNode3 = newNode -> next;
    newNode -> next = newNode3 -> next;
    newNode2 = newNode3 ->next;
    newNode2 ->prev = newNode;
    free(newNode3);
}

void Delete(){
    struct Node *newNode = head;
    struct Node *newNode2;
        head = newNode -> next;
        newNode2 = newNode -> next;
        head = newNode2;
        newNode2 -> prev = head;
        free(newNode);
    return;
}

bool isEmpty(){
    return head == NULL;
}

void Details(){
    printf("Cost Of Operations :-\n");
    printf("Accessing Element O(n)\n");
    printf("Inserting Element O(n)\n");
    printf("Deleting Element O(n)\n");
    printf("Searching for Element O(n)\n");
}

bool Search(int data){
    struct Node *newNode = head;
    while (newNode != NULL){
        if(newNode->data == data)
            return true;
        newNode = newNode -> next;
    }
    return false;
}

void SwapNodes(struct Node ** newNode){
    struct Node *newNode2 = NULL;
    struct Node *newNode3 = *newNode;
    newNode2 = newNode3 -> prev;
    newNode3 -> prev = newNode3 -> next;
    newNode3 -> next = newNode2;
}

void Reverse(){
   struct Node *newNode = NULL;
     struct Node *current = head;
     while (current !=  NULL)
     {
       newNode = current->prev;
       current->prev = current->next;
       current->next = newNode;
       current = current->prev;
     }


     if(newNode != NULL )
        head = newNode->prev;
}
