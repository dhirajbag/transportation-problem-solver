/*
____________________________________________________________________________________
     Student:
        Name: DHIRAJ BAG
        Roll: 001911001033
        B.E. in Information Technology
        2nd Year, 2nd Semester
        Jadavpur University (SaltLake Campus)
        Email: dhirajbag.db@gmail.com
        Date: 19th Apr, 2021
    
    Assignment:
        Title: Assignments for S/W Engg Lab - from Dr. B. C. Dhara

        Problem 1.
         => solution: Program will find optimal solution for a transportation
         problem.
            - Handles unbalanced problem
            - Handles Maximization problem
            - Uses VAM to get the initial basic feasible solution
            - Handles degeneracy whenever required
            - Used U-V method / MODI method for further optimization and
                to reach the optimal solution.

        Assumptions:
            - All the cost, demand, supply - all are non negetive integers

    Compiling and running:
        To compile use:
            gcc main.c -o main
        
        To run use:
            ./main

        Further instructions will be provided at runtime.
____________________________________________________________________________________
*/




#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define Epsilon 0.001  /* A very small value for handling degeneracy */
#define Zero 0.0000001 /* An extremely small value for comparing double values against zero */

int Round(double d){ /*Creating own Round() function to avoid math.h library */ 
    return (int)(d + 0.5);
}


/* node : used to represent a cell with row_index, col_index and optional value */
typedef struct node
{
    int row, col, data;
} node;


/*
Helper Functions implemented:

- void makeNegetive(int **cost, int num_rows, int num_cols) : multiplies -1 to all the entries  in cost matrix (for handling maximization problem)

- void check_and_balance(int ***costPtr, int **demandPtr, int **supplyPtr, int *num_rowsPtr, int *num_colsPtr) : Checks whether the problem is an unbalanced one.
        If it is unbalanced, makes all the necessary changes - modifies cost matrix, supply and demand array, numRows, numCols

- int penalty(cost, r1, r2, c1, c2) : returns the penalty (difference between minimum and 2nd minimum) within submatrix cost[r1:r2 :: c1:c2]
            agruments r1, r2, c1, c2 are such that the submatrix represents either a row or a column

- node findNextAllocated( A, r1, r2, c1, c2, direction) : finds the closest(next) allocated cell(node) in the specified direction(+1 or -1)
           within the submatrix A[r1:r2 :: c1:c2]  - r1,r2,c1,c2 set as such that it represents a sub column or a sub row

- int depth_first_search(start, current, direction, visited, members, topPtr, A, num_rows, num_cols) : visits nodes following some rules:
         - Next node must be an allocated neighbour of current node
         - Next node must be either to the right, left or straight(up) direction - cannot be in backward (down) direction
         - Next node must be unvisited
         - If node encountered is start node, dfs() ends returning 1
         - If no such next node is found, dfs() ends returning 0
       Directions of moving:
            1 reperesent left -> right direction
            2 represents up -> down direction
            3 represents right -> left direction
            4 represents down to up direction

- void printMatrixRounded( A, r1, r2, c1, c2) : prints the elements (double) of submatrix A[r1:r2 :: c1:c2] rounded to nearest integer


Some other helper functions impelemented:

    - int* allocate_row(int n) : allocates an array of n integers all set to zero

    - int** allocate_matrix_int(int num_rows, int num_cols) : allocates a matrix of integer elements all set to 0

    - double** allocate_matrix_double(int num_rows, int num_cols) : allocates a matrix of double elements all set to 0.0

    - void deallocate_matrix_int(int **matrix, int num_rows, int num_cols) : deallocates the integer matrix pointed by 'matrix'

    - void deallocate_matrix_double(double **matrix, int num_rows, int num_cols) : deallocates the double matrix pointed by 'matrix'

    - void input_row_int(int *row, int n) : takes n integers as input (in the form of a row) and stores them in 'row'

    - void input_matrix_int(int **matrix, int num_rows, int num_cols) : takes num_rows*num_cols number of integers as input
        (in the form of a matrix) and stores them in 'matrix'

    - void set_all_to(int* row, int n, int value) : sets values of all the n elements in the array 'row' to 'value'

 */


/* Function prototypes */
void makeNegetive(int **cost, int num_rows, int num_cols);
void check_and_balance(int ***costPtr, int **demandPtr, int **supplyPtr, int *num_rowsPtr, int *num_colsPtr);
int penalty(int **cost, int r1, int r2, int c1, int c2);
node findNextAllocated(double **A, int r1, int r2, int c1, int c2, int direction);
int depth_first_search(node start, node current, int direction, int *visited, node *members, int *topPtr, double **A, int num_rows, int num_cols);
void printMatrixRounded(double **A, int r1, int r2, int c1, int c2);

int* allocate_row(int n);
int** allocate_matrix_int(int num_rows, int num_cols);
double** allocate_matrix_double(int num_rows, int num_cols);
void deallocate_matrix_int(int **matrix, int num_rows, int num_cols);
void deallocate_matrix_double(double **matrix, int num_rows, int num_cols);

void input_row_int(int *row, int n);
void input_matrix_int(int **matrix, int num_rows, int num_cols);

void set_all_to(int* row, int n, int value);
/*__________________________________*/

int main()
{
    printf("Transportation Problem Solver: (- Dhiraj Bag, Roll: 001911001033) \n");
    printf("\nEach row represents a source, ie. num(sources)=num(rows)\n");
    printf("Each column represents a destination, ie. num(destinations)=num(columns)\n");

    int num_rows, num_cols;
    printf("\nEnter num(Rows) <space> num(Columns): ");
    scanf("%d %d", &num_rows, &num_cols);

    int **cost = allocate_matrix_int(num_rows, num_cols);

    printf("\nEnter Cost matrix(%dx%d) <in rows and columns> :\n", num_rows, num_cols);

    input_matrix_int(cost, num_rows, num_cols);

    int *demand = allocate_row(num_cols);
    int *supply = allocate_row(num_rows);

    printf("\nEnter Demands in the form of a Row(%d) : ", num_cols);
    input_row_int(demand, num_cols);

    printf("Enter Supply capacities in the form of a Row(%d) : ", num_rows);
    input_row_int(supply, num_rows);

    int option;

    printf("\nEnter corresponding option number: \n\t1.Minimization\n\t2.Maximization\n\t: ");
    scanf("%d", &option);

    int isMaximization = (option == 2) ? 1 : 0;

    if (isMaximization)
        makeNegetive(cost, num_rows, num_cols);


    int original_num_rows = num_rows;
    int original_num_cols = num_cols;

    check_and_balance(&cost, &demand, &supply, &num_rows, &num_cols);

    /*____Vogel's Approximation Method______*/

    double **allocation = allocate_matrix_double(num_rows, num_cols);


    int isRowCrossed[num_rows];
    int isColCrossed[num_cols];

    set_all_to(isRowCrossed, num_rows, 0);
    set_all_to(isColCrossed, num_cols, 0);


    int row_penalty[num_rows], col_penalty[num_cols];

    int numAllocation = 0;

    while (numAllocation < (num_rows + num_cols - 1))
    {
        int i;
        for (i = 0; i < num_rows; i++)
        {
            if (!isRowCrossed[i])
                row_penalty[i] = penalty(cost, i, i, 0, num_cols - 1);
        }

        int j;
        for (j = 0; j < num_cols; j++)
        {
            if (!isColCrossed[j])
                col_penalty[j] = penalty(cost, 0, num_rows - 1, j, j);
        }

        int max_penalty_row_index = -1, row_max_penalty_val = INT_MIN;
        int max_penalty_col_index = -1, col_max_penalty_val = INT_MIN;

        for (i = 0; i < num_rows; i++)
        {
            if (!isRowCrossed[i])
            {
                if (row_penalty[i] > row_max_penalty_val)
                {
                    row_max_penalty_val = row_penalty[i];
                    max_penalty_row_index = i;
                }
            }
        }

        for (j = 0; j < num_cols; j++)
        {
            if (!isColCrossed[j])
            {
                if (col_penalty[j] > col_max_penalty_val)
                {
                    col_max_penalty_val = col_penalty[j];
                    max_penalty_col_index = j;
                }
            }
        }

        /*Cell (a_row, a_col) is to be allocated next */
        int a_row = -1;
        int a_col = -1;
        int min_cost = INT_MAX;

        if (max_penalty_col_index == -1 && max_penalty_row_index == -1)
            break;

        else if (max_penalty_col_index == -1 || (max_penalty_row_index != -1 && row_max_penalty_val >= col_max_penalty_val))
        {
            a_row = max_penalty_row_index;

            int j;
            for (j = 0; j < num_cols; j++)
            {
                if (!isColCrossed[j])
                {
                    if (cost[max_penalty_row_index][j] < min_cost)
                    {
                        a_col = j;
                        min_cost = cost[max_penalty_row_index][j];
                    }
                }
            }
        }
        else
        {
            a_col = max_penalty_col_index;

            int i;
            for (i = 0; i < num_rows; i++)
            {
                if (!isRowCrossed[i])
                {
                    if (cost[i][max_penalty_col_index] < min_cost)
                    {
                        a_row = i;
                        min_cost = cost[i][max_penalty_col_index];
                    }
                }
            }
        }

        if (a_row == -1 || a_col == -1) /* No more such cell is found - ie, all demands are fulfilled - initial solution found */
            break;

        if (supply[a_row] < demand[a_col])
        {
            allocation[a_row][a_col] = supply[a_row] * 1.0;
            demand[a_col] -= supply[a_row];
            supply[a_row] = 0;

            isRowCrossed[a_row] = 1;
        }
        else
        {
            allocation[a_row][a_col] = demand[a_col] * 1.0;
            supply[a_row] -= demand[a_col];
            demand[a_col] = 0;
            isColCrossed[a_col] = 1;
        }

        numAllocation++; /* Cell(a_row, a_col) is allocated */
    }

    /*Initial solution by VAM is found*/

    if (numAllocation != num_rows + num_cols - 1)
        printf("\nNote: Initial solution by VAM is degenerate! \n");

    printf("\nInitial Allocation is: \n");
    printMatrixRounded(allocation, 0, num_rows - 1, 0, num_cols - 1);



    /*________MODI / U-V Method_________________________*/

    while (1)
    {
        printf("\nApplying U-V method on allocation: \n");
        printMatrixRounded(allocation, 0, num_rows - 1, 0, num_cols - 1);
        puts("");

        /*Degeneracy Test*/
        int num_allocated = 0;
        int i, j;
        for (i = 0; i < num_rows; i++)
        {
            for (j = 0; j < num_cols; j++)
            {
                if (allocation[i][j] > Zero)
                    num_allocated++;
            }
        }

        if (num_allocated != num_rows + num_cols - 1)
        {
            printf("Note: U-V Method received degenerate solution. Handling degeneracy ...\n");

            int remaining = num_rows + num_cols - 1 - num_allocated;

            for (i = 0; i < num_rows; i++)
            {
                for (j = 0; j < num_cols; j++)
                {
                    if (allocation[i][j] <= Zero)
                    {
                        allocation[i][j] = Epsilon;
                        remaining--;
                    }

                    if (remaining == 0)
                        break;
                }
            }

            /* Degeneracy Handled */
        }

        /* Calculating values for U and V */

        int U[num_rows], V[num_cols];

        /* U[i] or V[i] set to INT_MAX means that it is still not calculated */
        set_all_to(U, num_rows, INT_MAX);
        set_all_to(V, num_cols, INT_MAX);

        U[0] = 0;
        int num_calculated = 1;

        while (num_calculated < num_rows + num_cols)
        {
            int i, j;

            for (i = 0; i < num_rows; i++)
            {
                for (j = 0; j < num_cols; j++)
                {
                    if (allocation[i][j] > Zero)
                    {
                        if (V[j] == INT_MAX && U[i] != INT_MAX)
                        {
                            V[j] = cost[i][j] - U[i];
                            num_calculated++;
                        }
                        else if (U[i] == INT_MAX && V[j] != INT_MAX)
                        {
                            U[i] = cost[i][j] - V[j];
                            num_calculated++;
                        }
                    }
                }
            }
        }

        /* Findling the unallocated cell(i=m_row, j=m_col) with maximum opportunity_cost: U[i]*V[j] - cost[i][j] */
        int max_opportunity = INT_MIN, m_row = -1, m_col = -1;

        for (i = 0; i < num_rows; i++)
        {
            for (j = 0; j < num_cols; j++)
            {
                if (allocation[i][j] <= Zero)
                {
                    if (U[i] + V[j] - cost[i][j] > max_opportunity)
                    {
                        max_opportunity = U[i] + V[j] - cost[i][j];
                        m_row = i;
                        m_col = j;
                    }
                }
            }
        }

        if (max_opportunity <= 0) /* Every cell has opportunity_cost <= 0 : Optimal solution found */
        {
            printf("\n\nOptimal Solution Reached.\n");
            if (max_opportunity == 0)
                printf("Alternate solution is also possible.\n");
            break;
        }
        else /* There exists at least one cell with opportunity_cost > 0 : Optimal solution not reached */
        {
            /* Forming loop from maximum opportunity_cost cell(m_row, m_col) */
            printf("Finding loop starting from cell (%d, %d) :\n", m_row + 1, m_col + 1);

            node start = {m_row, m_col, -1};
            node members[num_rows * num_cols];
            int top = -1;
            /*
            * start : starting node (unallocated) from which loop formation starts
            * members[] array will contain the nodes that forms the final loop
            * nodes will be stored in reverse order - start node will be the last element in members[] array
            * top : points to the current last element in members array
            */

            /* Temporarily making start vertex allocated, so that DFS() can find it */
            allocation[start.row][start.col] = 1.0;

            int found = 0;

            /* findNextAllocated() finds the closest next allocated cell in a column/row, in the specified direction (+1 or -1) */

            /* start node may have 4 allocated neighbours - left, right, up, down */
            node left = findNextAllocated(allocation, start.row, start.row, 0, start.col - 1, -1);
            node right = findNextAllocated(allocation, start.row, start.row, start.col + 1, num_cols - 1, +1);
            node up = findNextAllocated(allocation, 0, start.row - 1, start.col, start.col, -1);
            node down = findNextAllocated(allocation, start.row + 1, num_rows - 1, start.col, start.col, +1);

            /*
            * visited[] array tracks whether a node is already visited or not
            * node(row_index, col_index) maps to visited[i] by the rule, i = row_index*num_cols + col_index
            * initially all the nodes are unvisited
            */
            int visited[num_rows * num_cols];
            for (i = 0; i < num_rows * num_cols; i++)
                visited[i] = 0;

            /* Directions of moving:
                1 reperesent left -> right direction
                2 represents up -> down direction
                3 represents right -> left direction
                4 represents down -> up direction
             */

            /*
            * depth_first_search() visits nodes following some rules:
            *     - Next node must be an allocated neighbour of current node
            *     - Next node must be either to the right, left or straight(up) direction - cannot be in backward (down) direction
            *     - Next node must be unvisited
            *     - If node encountered is start node, dfs() ends returning 1
            *     - If no such next node is found, dfs() ends returning 0
            */

            if (left.col != -1)
                found = depth_first_search(start, left, 3, visited, members, &top, allocation, num_rows, num_cols); /* Moving from start to left, ie direction = 3 */

            if (!found && right.col != -1)
                found = depth_first_search(start, right, 1, visited, members, &top, allocation, num_rows, num_cols); /* Moving from start to right, ie direction = 1 */

            if (!found && up.col != -1)
                found = depth_first_search(start, up, 4, visited, members, &top, allocation, num_rows, num_cols); /* Moving from start to up, ie direction = 4 */

            if (!found && down.col != -1)
                found = depth_first_search(start, down, 2, visited, members, &top, allocation, num_rows, num_cols); /* Moving from start to down, ie direction = 2 */

            if (found)
                printf("Loop is formed\n");
            else
                printf("Error: couldn't form the loop.\n"); /* Won't happen since degeneracy is handled */

            /* Restoring the allocation value of start vertex - deallocated */
            allocation[start.row][start.col] = 0.0;

            members[++top] = start;

            int sign = 1;
            node minAllocatedNode = {-1, -1, -1}; /* Captures the minimum allocation node from the loop with sign = 0 */
            double minAllocation = (double)INT_MAX;

            for (i = top; i >= 0; i--)
            {
                node curr = members[i];
                int r = curr.row, c = curr.col;

                if (sign == 0 && allocation[r][c] < minAllocation)
                {
                    minAllocatedNode.row = r;
                    minAllocatedNode.col = c;
                    minAllocation = allocation[r][c];
                }

                sign ^= 1;
            }

            /*
            * Subtracting minAllocation from nodes with sign = 0
            * Adding minAllocation to nodes with sign 1
            */

            sign = 1;
            for (i = top; i >= 0; i--)
            {
                node curr = members[i];
                int r = curr.row, c = curr.col;

                if (sign == 1)
                {
                    allocation[r][c] += minAllocation;
                }
                else
                    allocation[r][c] -= minAllocation;

                sign ^= 1;
            }
        }
    }

    /* Optimal solution was found */

    printf("\n\n\n => Final Allocation (Optimal) is: \n");
    printMatrixRounded(allocation, 0, original_num_rows - 1, 0, original_num_cols - 1);

    /* Value of Optimized Cost/Profit */
    int total = 0;

    int i, j;
    for(i=0; i<original_num_rows; i++){
        for(j=0; j<original_num_cols; j++){

            int allotment = Round(allocation[i][j]);
            total += allotment*cost[i][j];
        }
    }

    if(isMaximization)
        total *= -1;

    printf("\n => Optimal value of cost (or, profit if maximization) = %d \n", total);


    /*Deallocation*/
    free(demand);
    free(supply);
    deallocate_matrix_double(allocation, num_rows, num_cols);
    deallocate_matrix_int(cost, num_rows, num_cols);
    /*________________________________________________________*/

    printf("\nYou have reached the end of the program. Press enter to exit.");
    getchar(); getchar();

    return 0;
}


void makeNegetive(int **cost, int num_rows, int num_cols){
    int i, j;

    for(i=0; i<num_rows; i++){
        for(j=0; j<num_cols; j++){
            cost[i][j] *= -1;
        }
    }
}

void check_and_balance(int ***costPtr, int **demandPtr, int **supplyPtr, int *num_rowsPtr, int *num_colsPtr){

    int *oldSupply = *supplyPtr;
    int *oldDemand = *demandPtr;
    int **oldCost = *costPtr;
    int m = *num_rowsPtr;
    int n = *num_colsPtr;

    int i, j;

    int total_supply=0, total_demand=0;

    for(i=0; i< m; i++)
        total_supply += (*supplyPtr)[i];

    for(i=0; i< n; i++)
        total_demand += (*demandPtr)[i];

    if(total_supply < total_demand){ /*Add an extra supply row */

        printf("\nBalancing: Adding a dummy row ...\n");

        int *newSupply = (int*) malloc( (m+1)*sizeof(int));
        for(i=0; i<m; i++)
            newSupply[i] = oldSupply[i];

        newSupply[m] = total_demand - total_supply;

        free(oldSupply);
        *supplyPtr = newSupply;

        int **newCost = allocate_matrix_int(m+1, n); /* By default, sets all to zero */
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                newCost[i][j] = oldCost[i][j];
            }
        }
        deallocate_matrix_int(oldCost, m, n);

        *costPtr = newCost;

        *num_rowsPtr = m+1;
    }
    else if (total_demand < total_supply){ /*Add an extra column */

        printf("\nBalancing: Adding a dummy column ...\n");

        int *newDemand = (int*) malloc((n+1)*sizeof(int));
        for(i=0; i<n; i++)
            newDemand[i] = oldDemand[i];

        newDemand[n] = total_supply - total_demand;

        free(oldDemand);
        *demandPtr = newDemand;

        int **newCost = allocate_matrix_int(m, n+1);  /* By default, sets all to zero */
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                newCost[i][j]=oldCost[i][j];
            }
        }
        deallocate_matrix_int(oldCost, m, n);

        *costPtr = newCost;

        *num_colsPtr = n+1;

    }

}


int penalty(int **cost, int r1, int r2, int c1, int c2)
{
    int min0 = INT_MAX, min1 = INT_MAX;

    int i, j;
    for (i = r1; i <= r2; i++)
    {
        for (j = c1; j <= c2; j++)
        {
            if (cost[i][j] <= min0)
            {
                min1 = min0;
                min0 = cost[i][j];
            }
            else if (cost[i][j] < min1)
            {
                min1 = cost[i][j];
            }
        }
    }

    return (min1 - min0);
}

void printMatrixRounded(double **A, int r1, int r2, int c1, int c2)
{
    int i, j;
    for (int i = r1; i <= r2; i++)
    {
        for (int j = c1; j <= c2; j++)
        {
            int val = Round(A[i][j]);
            printf("%d\t", val);
        }
        printf("\n");
    }
}

node findNextAllocated(double **A, int r1, int r2, int c1, int c2, int direction)
{
    node tmp = {-1, -1, -1};
    int i, j;
    if (direction == 1)
    {
        for (i = r1; i <= r2; i++)
        {
            for (j = c1; j <= c2; j++)
            {
                if (A[i][j] > Zero)
                {
                    tmp.row = i;
                    tmp.col = j;
                    return tmp;
                }
            }
        }
    }
    else if (direction == -1)
    {
        for (i = r2; i >= r1; i--)
        {
            for (j = c2; j >= c1; j--)
            {
                if (A[i][j] > Zero)
                {
                    tmp.row = i;
                    tmp.col = j;
                    return tmp;
                }
            }
        }
    }

    return tmp;
}

int depth_first_search(node start, node current, int direction, int *visited, node *members, int *topPtr, double **A, int num_rows, int num_cols)
{

    if (current.col == start.col && current.row == start.row)
        return 1;

    else if (visited[num_cols * current.row + current.col] == 1)
        return 0;

    else
    {
        visited[num_cols * current.row + current.col] = 1;
        int found = 0;

        node left, right, up;
        int left_dir, right_dir, up_dir;

        if (direction == 1)
        {
            left_dir = 4;
            left = findNextAllocated(A, 0, current.row - 1, current.col, current.col, -1);

            right_dir = 2;
            right = findNextAllocated(A, current.row + 1, num_rows - 1, current.col, current.col, +1);

            up_dir = 1;
            up = findNextAllocated(A, current.row, current.row, current.col + 1, num_cols - 1, +1);
        }
        else if (direction == 2)
        {
            left_dir = 1;
            left = findNextAllocated(A, current.row, current.row, current.col + 1, num_cols - 1, +1);

            right_dir = 3;
            right = findNextAllocated(A, current.row, current.row, 0, current.col - 1, -1);

            up_dir = 2;
            up = findNextAllocated(A, current.row + 1, num_rows - 1, current.col, current.col, +1);
        }
        else if (direction == 3)
        {
            right_dir = 4;
            right = findNextAllocated(A, 0, current.row - 1, current.col, current.col, -1);

            left_dir = 2;
            left = findNextAllocated(A, current.row + 1, num_rows - 1, current.col, current.col, +1);

            up_dir = 3;
            up = findNextAllocated(A, current.row, current.row, 0, current.col - 1, -1);
        }
        else if (direction == 4)
        {
            right_dir = 1;
            right = findNextAllocated(A, current.row, current.row, current.col + 1, num_cols - 1, +1);

            left_dir = 3;
            left = findNextAllocated(A, current.row, current.row, 0, current.col - 1, -1);

            up_dir = 4;
            up = findNextAllocated(A, 0, current.row - 1, current.col, current.col, -1);
        }

        int last_dir;

        if (left.col != -1)
        {
            found = depth_first_search(start, left, left_dir, visited, members, topPtr, A, num_rows, num_cols);
            last_dir = left_dir;
        }
        if ( !found && right.col != -1)
        {
            found = depth_first_search(start, right, right_dir, visited, members, topPtr, A, num_rows, num_cols);
            last_dir = right_dir;
        }
        if ( !found && up.col != -1)
        {
            found = depth_first_search(start, up, up_dir, visited, members, topPtr, A, num_rows, num_cols);
            last_dir = up_dir;
        }

        if (!found)
        {
            return 0;
        }
        else
        {
            if (last_dir != direction) /* ie, it is a 90 degree turning point - it will be a member node */
                members[++(*topPtr)] = current;

            return 1;
        }
    }
}


int** allocate_matrix_int(int num_rows, int num_cols){
    int **matrix = (int**) malloc(num_rows*sizeof(int*));

    int i, j;
    for(i=0; i<num_rows; i++){
         matrix[i] = (int*) malloc(num_cols*sizeof(int));

        for(j=0; j<num_cols; j++)
            matrix[i][j] = 0;
    }

    return matrix;
}

double** allocate_matrix_double(int num_rows, int num_cols){
    double **matrix = (double**) malloc(num_rows*sizeof(int*));

    int i, j;
    for(i=0; i<num_rows; i++){
        matrix[i] = (double*) malloc(num_cols*sizeof(double));

        for(j=0; j<num_cols; j++)
            matrix[i][j] = 0.0;
    }

    return matrix;
}

int* allocate_row(int n){
    int *row = (int*)malloc(n*sizeof(int));
    int i;
    for(i=0; i<n; i++)
        row[i] = 0;

    return row;
}

void deallocate_matrix_int(int **matrix, int num_rows, int num_cols){
    int i;
    for(i=0; i<num_rows; i++){
        free(matrix[i]);
    }

    free(matrix);
}

void deallocate_matrix_double(double **matrix, int num_rows, int num_cols){
    int i;
    for(i=0; i<num_rows; i++){
        free(matrix[i]);
    }

    free(matrix);
}

void input_row_int(int *row, int n){
    int i;
    for(i=0; i<n; i++)
        scanf("%d", &row[i]);
}

void input_matrix_int(int **matrix, int num_rows, int num_cols){
    int i,j;
    for(i=0; i<num_rows; i++){
        input_row_int(matrix[i], num_cols);
    }
}



void set_all_to(int* row, int n, int value){
    int i;
    for(i=0; i<n; i++)
        row[i] = value;
}
