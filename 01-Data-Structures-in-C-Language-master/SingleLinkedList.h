// IMPLEMENTATION LINKED LIST IN C

/**
   |__|__| ->  |__|__| ->  |__|__| ->  |__|__| -> NULL

*/


/** A Linked List Node **/
struct Node{

    //space to store data
    int data;
    //link to the next node
    struct Node *next;
};

/** memory location to store the head position **/
struct Node *head = NULL;


/**               Insert(int)
    Function to add a node at the beginning of Linked List.
    this function have single parameter,
    takes data that want to be sort in this parameter
*/
void Insert(int data){
    // Allocate memory for new node
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    // store data in node
    newNode -> data = data;
    // referring the next node to the head
    newNode -> next = head;
    // referring the head to the new node to make it first node in accessing
    head = newNode;

    return; //END FUNCTION
}


/**               Insert(int,int)   :   OverLoaded Function
    Function to add a node at nth position in Linked List.
    this function have dual parameter,
    takes data that want to be sort in the first parameter,
    takes position that want to sort in in the second parameter,

*/
void Insert(int data, int n){
    // Allocate memory for new node
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    // store data in node
    newNode -> data = data;
    //Condition for checking if the position is the position of the head
    if(n == 1){
        // referring the next node to the head
        newNode -> next = head;
        // referring the head to the new node to make it first node in accessing
        head = newNode;
        return; //END FUNCTION
   }
    /**
        when the position is not beginning of the Linked List:
    */
    // this node for store the position of head to make our process without  destroy our head
    struct Node *newNode2 = head;
    //loop to walk to the node that in it's next we will put our newNode
    for(int i = 0; i<n-2; i++){
        newNode2 = newNode2 -> next;
    }
    // putting new node in nth position
    newNode -> next = newNode2 -> next;
    newNode2 -> next = newNode;
    return;
}

/**
            Delete()
    Function to delete the first node in Linked List
    has void parameter
*/
void Delete(){
    // referring allocate  node to the head
    struct Node *newNode = head;
    // referring head to  next node
    head = newNode -> next;
    //delete node form memory
    free(newNode);
    return; //END
}

/**
            Delete(int)     OverLoaded
    Function to delete nth position node in Linked List
    has single parameter to select position
*/
void Delete(int n){
    // referring allocate  node to the head
    struct Node *newNode = head;
    //Condition for checking if the position is the position of the head
    if (n == 1){
        // referring head to  next node
        head = newNode -> next;
        //delete node form memory
        free(newNode);
        return; //END
   }

    /**
        when the position is not beginning of the Linked List:
    */
    // this node for store the position of head to make our process without  destroy our head
    for(int i =0; i < n-2; i++)
        newNode =  newNode -> next;
    //fixing links in Linked List
    struct Node* newNode2 = newNode -> next;
    newNode -> next = newNode2 -> next;
    //Deleting node from memory
    free(newNode2);

}
/**
        ViewData()
    Function as it's name
*/
void ViewData(){
    struct Node *newNode = head;
    while (newNode != NULL){
        printf("%d ",newNode->data);
        newNode = newNode -> next;
    }
    printf("\n");
}

/**
        Details()
    Function to view complexity of each process
*/

void Details(){
    printf("Cost Of Operations :-\n");
    printf("Accessing Element O(n)\n");
    printf("Inserting Element O(n)\n");
    printf("Deleting Element O(n)\n");
    printf("Searching for Element O(n)\n");
}



void Reverse(){
    struct Node *current, *prev, *next;
    current = head;
    prev = NULL;
    while(current != NULL){
        next = current -> next;
        current -> next = prev;
        prev = current;
        current = next;
    }
    head = prev;

}

bool isEmpty(){
    return head == NULL;
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

int Index(int n){
    struct Node *newNode = head;
    for(int i =0; i < n-1; i++)
        newNode =  newNode -> next;
    return newNode -> data;
}
