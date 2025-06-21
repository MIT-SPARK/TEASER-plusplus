/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>

#include "teaser/ply_io.h"
#include "tinyply.h"

// Internal datatypes for storing ply vertices
struct float3 {
  float x, y, z;
};
struct double3 {
  double x, y, z;
};

int teaser::PLYReader::read(const std::string& file_name, teaser::PointCloud& cloud) {
  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;

  try {
    file_stream.reset(new std::ifstream(file_name, std::ios::binary));

    if (!file_stream || file_stream->fail()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      return -1;
    }

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    std::shared_ptr<tinyply::PlyData> vertices;

    try {
      vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception& e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
      return -1;
    }

    file.read(*file_stream);

    if (vertices) {
      std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
      if (vertices->t == tinyply::Type::FLOAT32) {
        std::vector<float3> verts_floats(vertices->count);
        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::memcpy(verts_floats.data(), vertices->buffer.get(), numVerticesBytes);
        for (auto& i : verts_floats) {
          cloud.push_back({i.x, i.y, i.z});
        }
      }
      if (vertices->t == tinyply::Type::FLOAT64) {
        std::vector<double3> verts_doubles(vertices->count);
        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::memcpy(verts_doubles.data(), vertices->buffer.get(), numVerticesBytes);
        for (auto& i : verts_doubles) {
          cloud.push_back(
              {static_cast<float>(i.x), static_cast<float>(i.y), static_cast<float>(i.z)});
        }
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}

int teaser::PLYWriter::write(const std::string& file_name, const teaser::PointCloud& cloud,
                             bool binary_mode) {
  // Open file buffer according to binary mode
  std::filebuf fb;
  if (binary_mode) {
    fb.open(file_name, std::ios::out | std::ios::binary);
  } else {
    fb.open(file_name, std::ios::out);
  }

  // Open output stream
  std::ostream outstream(&fb);
  if (outstream.fail()) {
    std::cerr << "Failed to open " << file_name << std::endl;
    return -1;
  }

  // Use tinyply to write to ply file
  tinyply::PlyFile ply_file;
  std::vector<float3> temp_vertices;
  for (auto& i : cloud) {
    temp_vertices.push_back({i.x, i.y, i.z});
  }
  ply_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, temp_vertices.size(),
      reinterpret_cast<uint8_t*>(temp_vertices.data()), tinyply::Type::INVALID, 0);
  ply_file.write(outstream, binary_mode);

  return 0;
}
