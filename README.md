# Kami Solver

This tool aims to automatically solve puzzles from the game [Kami](http://www.stateofplaygames.com/work/kami/).

[Demo Solution](http://hexylena.github.io/kami-solver/)

## Process

- [x] Detect rectangles

    ![](./media/rect.png)

- [x] Figure out grid pattern
    - From above image, find median distance, use that.
- [x] Pick out a colour from inside of each square

    ![](./media/pick.png)

- [x] Accurately cluster those

    ![](./media/ex.png)

- [x] Build neighbour graph

    ![](./media/graph.png)

- [x] Solve graph (bruteforce)
