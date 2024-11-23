/*

 you can consider 
atomic
task
for
for schedule dynamic or static
reduction
critical
section
parallel


 */
 
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>
#include <fstream>
#include <cstdlib>  // For random number generation
#include <ctime>

struct Tablo {
    std::vector<int> tab;
    int size;

    // Constructor to initialize the vector with the given size
    Tablo(int size) : size(size) {
        if (size > 0) {
            tab.resize(size, 0);  // Properly resize the vector to the given size
        }
    }

    // Function to print the array values
    void printArray() const {
        std::cout << "---- Array of size " << size << " ----\n";
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            #pragma omp critical
            std::cout << tab[i] << " ";
        }
        std::cout << "\n";
    }
};

void psum_montee(Tablo &source, Tablo &destination) {
    // Copy source to the second half of the destination
    #pragma omp parallel for
    for (int i = 0; i < source.size; i++) {
        destination.tab[destination.size / 2 + i] = source.tab[i];
    }

    // Up-sweep phase of prefix sum
    int depth = static_cast<int>(log2(source.size));
   // #pragma omp parallel
    {
    	#pragma omp parallel for
        for (int l = depth; l > 0; l--) {
            #pragma omp parallel for
            for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
                destination.tab[j] = destination.tab[2 * j] + destination.tab[2 * j + 1];
            }
        }
    }
}

void psum_descente(Tablo &a, Tablo &b) {
    b.tab[1] = 0;
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int l = 2; l <= depth; l++) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            if (j % 2 == 0) {
                b.tab[j] = b.tab[j / 2];
            } else {
                b.tab[j] = b.tab[j / 2] + a.tab[j - 1];
            }
        }
    }
}

void psum_final(Tablo &a, Tablo &b) {
    int depth = static_cast<int>(log2(a.size));

    #pragma omp parallel for
    for (int j = 1 << (depth - 1); j <= (1 << depth) - 1; j++) {
        b.tab[j] += a.tab[j];
    }
}



void ssum_montee(const Tablo &source, Tablo &destination) {
    // Copy source into the reversed second half of the destination array
    #pragma omp parallel for
    for (int i = 0; i < source.size; i++) {
        destination.tab[destination.size - i - 1] = source.tab[i];
    }

    // Up-sweep phase
    int depth = static_cast<int>(log2(source.size));
    #pragma omp parallel for
    for (int l = depth; l > 0; l--) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            destination.tab[j] = destination.tab[2 * j] + destination.tab[2 * j + 1];
        }
    }
}

void ssum_descente(const Tablo &a, Tablo &b) {
    b.tab[1] = 0;
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int l = 2; l <= depth; l++) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            if (j % 2 == 0) {
               // #pragma omp critical
                b.tab[j] = b.tab[j / 2];
            } else {
               // #pragma omp critical
                b.tab[j] = b.tab[j / 2] + a.tab[j - 1];
            }
        }
    }
}

void ssum_final(const Tablo &a, Tablo &b) {
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int j = 1 << (depth - 1); j <= (1 << depth) - 1; j++) {
        b.tab[j] += a.tab[j];
    }
}

void pmax_montee(const Tablo &source, Tablo &destination) {
    // Copy source into the reversed second half of the destination array
    #pragma omp parallel for
    for (int i = 0; i < source.size / 2; i++) {
        destination.tab[destination.size / 2 + i] = source.tab[source.size - i - 1];
    }

    // Up-sweep phase
    int depth = static_cast<int>(log2(source.size / 2));
    #pragma omp parallel for
    for (int l = depth; l > 0; l--) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            destination.tab[j] = std::max(destination.tab[2 * j], destination.tab[2 * j + 1]);
        }
    }
}

void pmax_descente(const Tablo &a, Tablo &b) {
    b.tab[1] = std::numeric_limits<int>::min();
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int l = 2; l <= depth; l++) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            if (j % 2 == 0) {
              //  #pragma omp critical
                b.tab[j] = b.tab[j / 2];
            } else {
               // #pragma omp critical
                b.tab[j] = std::max(b.tab[j / 2], a.tab[j - 1]);
            }
        }
    }
}

void pmax_final(const Tablo &a, Tablo &b) {
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int j = 1 << (depth - 1); j <= (1 << depth) - 1; j++) {
        b.tab[j] = std::max(b.tab[j], a.tab[j]);
    }
}


void smax_montee(const Tablo &source, Tablo &destination) {
    // Copy the second half of source in reverse order to destination
    #pragma omp parallel for
    for (int i = 0; i < source.size / 2; i++) {
        destination.tab[destination.size - i - 1] = source.tab[source.size / 2 + i];
    }

    // Up-sweep phase
    int depth = static_cast<int>(log2(source.size / 2));
    #pragma omp parallel for
    for (int l = depth; l > 0; l--) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            destination.tab[j] = std::max(destination.tab[2 * j], destination.tab[2 * j + 1]);
        }
    }
}

void smax_descente(const Tablo &a, Tablo &b) {
    b.tab[1] = std::numeric_limits<int>::min();
    int depth = static_cast<int>(log2(a.size));

    #pragma omp parallel for
    for (int l = 2; l <= depth; l++) {
        #pragma omp parallel for
        for (int j = 1 << (l - 1); j <= (1 << l) - 1; j++) {
            if (j % 2 == 0) {
               // #pragma omp critical
                b.tab[j] = b.tab[j / 2];
            } else {
                //#pragma omp critical
                b.tab[j] = std::max(b.tab[j / 2], a.tab[j - 1]);
            }
        }
    }
}

void smax_final(const Tablo &a, Tablo &b) {
    int depth = static_cast<int>(log2(a.size));
    #pragma omp parallel for
    for (int j = 1 << (depth - 1); j <= (1 << depth) - 1; j++) {
        b.tab[j] = std::max(b.tab[j], a.tab[j]);
    }
}



void make_max(const Tablo &q, const Tablo &pmax, const Tablo &smax, const Tablo &ssum, const Tablo &psum, Tablo &m) {
    Tablo ms(q.size), mp(q.size);

    #pragma omp parallel for
    for (int i = 0; i < q.size; i++) {
        ms.tab[i] = pmax.tab[pmax.size / 2 + i] - ssum.tab[ssum.size - i - 1];
        mp.tab[i] = smax.tab[smax.size - i - 1] - psum.tab[psum.size / 2 + i];
        m.tab[i] = ms.tab[i] + mp.tab[i] + q.tab[i];
    }
}

void createArray(Tablo &s) {
    std::cout << "Enter the size of the array: ";
    std::cin >> s.size;

    if (s.size <= 0) {
        std::cerr << "Invalid size.\n";
        return;
    }

    // Resize the array to the input size
    s.tab.resize(s.size);

    // Seed the random number generator
    srand(time(0));

    // Generate random numbers between 0 and 100
    #pragma omp parallel for
    for (int i = 0; i < s.size; ++i) {
        s.tab[i] = rand() % 100;  // Random number between 0 and 99
    }

    // Print generated values
    std::cout << "Generated array values: ";
    for (const int &val : s.tab) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}


void find_max(const Tablo &m, int &maxIndex, int &minIndex) {
    int value = std::numeric_limits<int>::min();
    int value_min = std::numeric_limits<int>::max();
    int value_max = std::numeric_limits<int>::min();

    std::vector<int> maxIndexA(m.size, std::numeric_limits<int>::min());
    std::vector<int> minIndexA(m.size, std::numeric_limits<int>::max());

    #pragma omp parallel for
    for (int i = 0; i < m.size; i++) {
        maxIndexA[i] = std::numeric_limits<int>::min();
        minIndexA[i] = std::numeric_limits<int>::max();
    }

    #pragma omp parallel for reduction(max: value)
    for (int i = 0; i < m.size; i++) {
        if (m.tab[i] > value) {
            value = m.tab[i];
            minIndexA[i] = i;
            maxIndexA[i] = i;
        } else if (m.tab[i] == value) {
            maxIndexA[i] = i;
        }
    }

    #pragma omp parallel for reduction(max: value_max) reduction(min: value_min)
    for (int i = 0; i < m.size; i++) {
        if ((maxIndexA[i] > value_max) && (value == m.tab[maxIndexA[i]])) {
            value_max = maxIndexA[i];
        }
        if ((minIndexA[i] < value_min) && (value == m.tab[minIndexA[i]])) {
            value_min = minIndexA[i];
        }
    }

    maxIndex = value_max;
    minIndex = value_min;
}


int main(int argc, char **argv) {
    
    
    Tablo source(0);
    createArray(source);
    std::cout << "SOURCE:\n";
    source.printArray();
#ifdef DEBUG
    std::cout << "SOURCE:\n";
    source.printArray();
#endif
    double start_time = omp_get_wtime(); 
    // Allocate temporary storage
    Tablo tmp(source.size * 2);

    // PSUM computation
    psum_montee(source, tmp);
   // tmp.printArray();
    
#ifdef DEBUG
    tmp.printArray();
#endif

    Tablo psum(source.size * 2);
    psum_descente(tmp, psum);
   // psum.printArray();
    
#ifdef DEBUG
    psum.printArray();
#endif

    psum_final(tmp, psum);
  //  psum.printArray();
    
#ifdef DEBUG
    psum.printArray();
#endif

    // SSUM computation
    //std::cout << "\nSSUM:\n";
    
#ifdef DEBUG
    std::cout << "\nSSUM:\n";
#endif
    ssum_montee(source, tmp);
   // tmp.printArray();
#ifdef DEBUG
    tmp.printArray();
#endif

    Tablo ssum(source.size * 2);
    ssum_descente(tmp, ssum);
    //ssum.printArray();
#ifdef DEBUG
    ssum.printArray();
#endif

    ssum_final(tmp, ssum);
  //  ssum.printArray();
#ifdef DEBUG
    ssum.printArray();
#endif

    // SMAX computation
    Tablo tmp2(source.size * 2);
   // std::cout << "\nSMAX:\n";
#ifdef DEBUG
    std::cout << "\nSMAX:\n";
#endif

    smax_montee(psum, tmp2);
  //  tmp2.printArray();
#ifdef DEBUG
    tmp2.printArray();
#endif

    Tablo smax(source.size * 2);
    smax_descente(tmp2, smax);
  //  smax.printArray();
#ifdef DEBUG
    smax.printArray();
#endif

    smax_final(tmp2, smax);
   // smax.printArray();
#ifdef DEBUG
    smax.printArray();
#endif

    // PMAX computation
   // std::cout << "\nPMAX:\n";
#ifdef DEBUG
    std::cout << "\nPMAX:\n";
#endif

    pmax_montee(ssum, tmp2);
   // tmp2.printArray();
#ifdef DEBUG
    tmp2.printArray();
#endif

    Tablo pmax(source.size * 2);
    pmax_descente(tmp2, pmax);
   // pmax.printArray();
#ifdef DEBUG
    pmax.printArray();
#endif

    pmax_final(tmp2, pmax);
   // pmax.printArray();
#ifdef DEBUG
    pmax.printArray();
#endif

    // Free temporary storage
    tmp2.tab.clear();

    // MAX computation
    //std::cout << "\nMAX:\n";
#ifdef DEBUG
    std::cout << "\nMAX:\n";
#endif

    Tablo m(source.size);
    make_max(source, pmax, smax, ssum, psum, m);
   // m.printArray();
   // std::cout << "\nRESULT:\n";
#ifdef DEBUG
    m.printArray();
    std::cout << "\nRESULT:\n";
#endif
    
    double end_time = omp_get_wtime();  // End time
    std::cout << "computation time: " << end_time - start_time << " seconds.\n";
    
    int maxIndex = -1, minIndex = -1;
    find_max(m, maxIndex, minIndex);

    std::cout << m.tab[minIndex];
#pragma omp parallel for
    for (int i = minIndex; i <= maxIndex; i++) {
        if (m.tab[i] == m.tab[minIndex]) {
            std::cout << " " << source.tab[i];
        } else {
            i = maxIndex + 1;
        }
    }
    std::cout << "\n";

    return 0;
}
