#pragma once

#include <fstream>
#include <string>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <ng-log/logging.h>

// Read a proto message from a file (.pb.txt = text format, .pb = binary).
template <typename T> T ReadProtoFromFile(const std::string &path) {
  LOG(INFO) << "Reading from " << path << " ...";
  T message;
  if (path.ends_with(".pb.txt")) {
    std::ifstream in(path);
    CHECK(in);
    google::protobuf::io::IstreamInputStream is(&in);
    CHECK(google::protobuf::TextFormat::Parse(&is, &message));
  } else if (path.ends_with(".pb")) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in);
    CHECK(message.ParseFromIstream(&in));
  } else {
    LOG(FATAL) << "Unsupported file extension: " << path;
  }
  LOG(INFO) << "Read " << message.ByteSizeLong() << " bytes";
  return message;
}

// Write a proto message to a file (.pb.txt = text format, .pb = binary).
template <typename T>
void WriteProtoToFile(const T &message, const std::string &path,
                      bool also_write_to_txt = false, bool quiet = false) {
  if (!quiet) {
    LOG(INFO) << "Writing to " << path << " ...";
  }
  if (path.ends_with(".pb.txt")) {
    std::ofstream out(path);
    CHECK(out) << "Failed to open file: " << path;
    google::protobuf::io::OstreamOutputStream os(&out);
    CHECK(google::protobuf::TextFormat::Print(message, &os));
  } else if (path.ends_with(".pb")) {
    std::ofstream out(path, std::ios::binary);
    CHECK(out) << "Failed to open file: " << path;
    CHECK(message.SerializeToOstream(&out));
    if (also_write_to_txt) {
      WriteProtoToFile(message, path + ".txt");
    }
  } else {
    LOG(FATAL) << "Unsupported file extension: " << path;
  }
  if (!quiet) {
    LOG(INFO) << "Written " << message.ByteSizeLong() << " bytes";
  }
}
