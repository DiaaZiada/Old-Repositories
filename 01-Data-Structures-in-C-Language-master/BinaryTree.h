#include <queue>

struct Node {
    int data;
    struct Node *left;
    struct Node *right;
};

struct Node *root = NULL;

bool isEmpty(){
    return(root == NULL);
}

bool Search(int data){
    struct Node *current = root;

    while(current != NULL){
        if(data == current -> data)
            return true;
        else
            if(data <= current -> data)
                current = current -> left;
            else
                current = current -> right;
    }
    return false;
}

int MinValue(){
    if(isEmpty()){
        printf("Empty Tree!");
        return -1;
    }
    struct Node *current = root;
    while (current -> left != NULL)
        current = current -> left;
    return current -> data;
}

int MaxValue(){
    if(isEmpty()){
        printf("Empty Tree!");
        return -1;
    }
    struct Node *current = root;
    while (current -> right != NULL)
        current = current -> right;
    return current -> data;
}

struct Node *CreatingNewNode(int data){
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode -> data = data;
    newNode -> left = newNode -> right = NULL;
    return newNode;
}


struct Node* Insert(struct Node* node, int key)
{
    if (node == NULL){
        return CreatingNewNode(key);;
    }
    if (key < node->data)
        node->left  = Insert(node->left, key);
    else if (key > node->data)
        node->right = Insert(node->right, key);
    return node;
}

void Insert(int data){
    root = Insert(root,data);
}


int HeightOfTree(){

    if(root == NULL)
        return 0;
    queue<struct Node*> Q;

    Q.push(root);
    int height = 0;

    while(true){
        int noNodes = Q.size();
        if(noNodes == 0)
            return height;
        height++;

        while(noNodes > 0){
            struct Node *newNode = Q.front();
            Q.pop();
            if(newNode -> left != NULL) Q.push(newNode -> left);
            if(newNode -> right != NULL) Q.push(newNode -> right);
            noNodes--;

        }
    }


}

void LevelOrderTraversal(struct Node *root){
    if(root == NULL)
        return;
    queue<struct Node*> Q;
    Q.push(root);

    while(!Q.empty()){
        struct Node *current = Q.front();
        cout<<current-> data<<" ";
        if(current->left != NULL) Q.push(current->left);
        if(current->right != NULL) Q.push(current->right);
        Q.pop();
    }
}

void LevelOrderTraversal(){
    LevelOrderTraversal(root);
}

void InorderTraversal(struct Node *root){
    if(root == NULL)
        return;
    InorderTraversal(root->left);
    printf("%d",root->data);
    InorderTraversal(root->right);
}

void InorderTraversal(){
    InorderTraversal(root);
}

void PostorderTraversal(struct Node *root){
    if(root == NULL)
        return;
    PostorderTraversal(root->left);
    PostorderTraversal(root->right);
    printf("%d",root->data);

}

void PostorderTraversal(){
    PostorderTraversal(root);
}

void PreorderTraversal(struct Node *root){
    if(root == NULL)
        return;
    printf("%d",root->data);
    PreorderTraversal(root->left);
    PreorderTraversal(root->right);

}

void PreorderTraversal(){
    PreorderTraversal(root);
}

struct Node *FindMin(struct Node *root){
    while(root -> left != NULL) root = root -> left;
    return root;
};

struct Node *FindMax(struct Node *root){
    while(root -> right != NULL) root = root -> right;
    return root;
};

struct Node *Delete(struct Node *root,int data){
    if(root == NULL)
        return root;
    else if(root -> data > data) root -> left = Delete(root -> left, data);
    else if(root -> data < data) root -> right = Delete(root -> right, data);
    else{

        if(root -> left == NULL && root -> right == NULL){
            delete root;
            root = NULL;
            return root;
        }
        else if(root -> left == NULL){
            struct Node *newNode = root;
            root = root -> right;
            delete newNode;
            return root;
        }
        else if(root -> right == NULL){
            struct Node *newNode = root;
            root = root -> left;
            delete newNode;
            return root;
        }
        else{
            struct Node *newNode = FindMin(root -> right);
            root -> data = newNode -> data;
            root -> right = Delete(root -> right, newNode -> data);
        }
    }
}


void Delete(int data){
    root = Delete(root,data);
}

void Details(){
}
