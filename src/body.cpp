/*---------------------------------------------------------------------------*\
 *
 *  minimmerflow
 *
 *  Copyright (C) 2015-2021 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of minimmerflow.
 *
 *  minimmerflow is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  minimmerflow is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with minimmerflow. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#include "body.hpp"

#include <bitpit_IO.hpp>

#include <vector>

using namespace bitpit;

namespace body {

struct BodyBox {
    double xMin;
    double yMin;
    double zMin;
    double xMax;
    double yMax;
    double zMax;
};

std::vector<BodyBox> body_boxes;

/*!
 * Initialize the bodies.
 */
void initialize()
{
    if (!config::root.hasSection("bodies")) {
        return;
    }

    for (const auto &bodyEntry : config::root["bodies"].getSections()) {
        const Config &bodyInfo = *bodyEntry.second;
        const std::string bodyShape = bodyInfo.get<std::string>("type", "");
        if (bodyShape == "box") {
            body_boxes.emplace_back();
            BodyBox &box = body_boxes.back();
            box.xMin = bodyInfo.get<double>("xMin");
            box.xMax = bodyInfo.get<double>("xMax");
            box.yMin = bodyInfo.get<double>("yMin");
            box.yMax = bodyInfo.get<double>("yMax");
            box.zMin = bodyInfo.get<double>("zMin");
            box.zMax = bodyInfo.get<double>("zMax");
        } else {
            throw std::runtime_error("Body shape \"" + bodyShape + "\" is not supported.");
        }
    }
}

/*!
 * Check if the given point is inside the fluid.
 *
 * \param point are the coordinates of the point
 * \result Return true if the given point is inside the fluid, false otherwise.
 */
bool isPointFluid(const std::array<double, 3> &point)
{
    for (const BodyBox &box : body_boxes) {
        if (point[0] < box.xMin || point[0] > box.xMax) {
            continue;
        } else if (point[1] < box.yMin || point[1] > box.yMax) {
            continue;
        } else if (point[2] < box.zMin || point[2] > box.zMax) {
            continue;
        }

        return false;
    }

    return true;
}

}
