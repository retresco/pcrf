
#include <iostream>

int main()
{
  char ch;
  unsigned offset = 0;
  while (std::cin.good()) {
    std::cin.get(ch);
    std::cout << offset++ << "\t" << ((ch <= ' ') ? ' ' : ch) << std::endl;
  }
}
