# conjunctions


Using [NASA/JPL's development ephemeris](https://naif.jpl.nasa.gov/naif/data.html) and the [Skyfield python module](https://rhodesmill.org/skyfield/) this script calculates when the 7 other planets of the solar system, plus the Moon and Pluto, are at their minimum angular separation. It was riginal created to better understand the "great conjunction" of Jupiter and Saturn on 2020-12-21.

The script calculates the separation on each day between a specified start and stop date, or across the entire available period in the JPL spice kernel file.  For the DE406 kernel specified this spans 3000 BCE to 3000 AD, nearly 2.9 million days.  Once those minima are found, those that are greater than the sum of the orbital inclinations of the pair are flitered out and the remainder are calculated from day to minute accuracy (recursively). The results are written to an Excel file, one tab per planet or Moon pair.  

With between 300 (Saturn-Uranus) and over 74k (Mercury-Moon) conjunctions to calculate at that minute resolution, it takes a while to run.  A 32-CPU cloud host  with 32 gig of memory running Ubuntu took about an hour to complete.
