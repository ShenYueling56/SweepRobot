/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "include/Util/selfDefine.h"

#if defined USE_FBOW
#include "Thirdparty/fbow/src/fbow.h"
#elif defined USE_DBOW3

#include "Thirdparty/DBow3/src/DBoW3.h"

#elif defined USE_DBOW2
#include"Thirdparty/DBoW2/DBoW2/FORB.h"
#include"Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#endif

#include "Converter.h"

namespace ORB_SLAM3 {

#if defined USE_FBOW
    typedef fbow::Vocabulary ORBVocabulary;
    typedef fbow::fBow BowVec;
    typedef fbow::fBow2 FeatVec;
#elif defined USE_DBOW3
    typedef DBoW3::Vocabulary ORBVocabulary;
    typedef DBoW3::BowVector BowVec;
    typedef DBoW3::FeatureVector FeatVec;
#elif defined USE_DBOW2
    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
    typedef DBoW2::BowVector BowVec;
    typedef DBoW2::FeatureVector FeatVec;
#endif

    class VOCProsessor {
    public:
        static void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out) {
            out.resize(plain.rows);

            for (int i = 0; i < plain.rows; ++i) {
                out[i] = plain.row(i);
            }
        }

        static float score(const ORBVocabulary *pVoc, BowVec bowV1, BowVec bowV2) {
#if defined USE_FBOW
            float si = bowV1.score(bowV1,bowV2);
#elif defined USE_DBOW2 || defined USE_DBOW3
            float si = pVoc->score(bowV1, bowV2);
#endif
            return si;
        }

        static unsigned int size(const ORBVocabulary *pVoc) {
            unsigned int s = 0;
#if defined USE_FBOW
            s = pVoc->leafSize();
#elif defined USE_DBOW2 || defined USE_DBOW3
            s = pVoc->size();
#endif
            return s;
        }

        static unsigned int size(const ORBVocabulary &voc) {
            unsigned int s = 0;
#if defined USE_FBOW
            s = voc.leafSize();
#elif defined USE_DBOW2 || defined USE_DBOW3
            s = voc.size();
#endif
            return s;
        }

        static void
        transform(const ORBVocabulary *pVoc, const cv::Mat &desc, BowVec &mBowVec, FeatVec &mFeatVec, int level) {
#if defined USE_FBOW
            pVoc->transform(desc,level,mBowVec,mFeatVec);
#elif defined USE_DBOW2 || defined USE_DBOW3
            std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(desc);
            pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, level);
#endif
        }
    };
} //namespace ORB_SLAM

#endif // ORBVOCABULARY_H
