//
//  particle.h
//  Alamo
//
//  Created by Maurizio Favaron on 27/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#ifndef particle_h
#define particle_h

class Particle {
    
private:
    
    double    age;
    double    x, y, z;    // Center position (m)
    double    u, v, w;    // Velocity
    double    Q;          // Mass
    double    T;          // Temperature
    double    sh, sz;     // Gaussian kernel horizontal and vertical sigmas
    
public:
    
    
    
};

#endif /* particle_h */
