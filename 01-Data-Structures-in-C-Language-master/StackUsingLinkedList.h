
#include "DoublyLinkedList.h"
int noData = 0;
void Push(int data){
    noData++;
    InsertAtTail(data);
}
void pop(){
    Delete(--noData);
    noData--;
}

void PrintStack(){
    Print();
}
void StackDetails(){

}




