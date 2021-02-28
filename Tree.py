class TreeNode:
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
    #打印树
class TreeNodeTools:
    def printf(self,root):
        print(root.val)
        if root.right:
            self.printf(root.right)
        if root.left:
            self.printf(root.left)
    #行序遍历建树
    def createTreeByrow(self,llist,i):
        if llist[i]=='null':
            return
        root=TreeNode(llist[i])
        if i*2+1<len(llist):
            root.left=self.createTreeByrow(llist,i*2+1)
        if i*2+2<len(llist):
            root.right=self.createTreeByrow(llist,i*2+2)
        return root
if __name__ == "__main__":
    ss=TreeNodeTools()
    root3=ss.createTreeByrow([5,4,8,11,'null',13,4,7,2,'null','null',5,1],0)
    ss.printf(root3)