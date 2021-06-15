#pragma once

// ============================================================================
// == ./BlobAllocators.hpp ==
// ==
// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
	// ============================================================================
	// == ./Array.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
		// ============================================================================
		// == ./macros.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		#if defined(__GNUC__)
		#    define LLAMA_INDEPENDENT_DATA _Pragma("GCC ivdep")
		#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
		#    define LLAMA_INDEPENDENT_DATA _Pragma("ivdep")
		#elif defined(__clang__)
		#    define LLAMA_INDEPENDENT_DATA _Pragma("clang loop vectorize(enable) interleave(enable) distribute(enable)")
		#elif defined(_MSC_VER)
		#    define LLAMA_INDEPENDENT_DATA __pragma(loop(ivdep))
		#else
		/// May be put in front of a loop statement. Indicates that all (!) data access
		/// inside the loop is indepent, so the loop can be safely vectorized. Example:
		/// \code{.cpp}
		///     LLAMA_INDEPENDENT_DATA
		///     for(int i = 0; i < N; ++i)
		///         // because of LLAMA_INDEPENDENT_DATA the compiler knows that a and b
		///         // do not overlap and the operation can safely be vectorized
		///         a[i] += b[i];
		/// \endcode
		#    define LLAMA_INDEPENDENT_DATA
		#endif

		#ifndef LLAMA_FN_HOST_ACC_INLINE
		#    if defined(__NVCC__)
		#        define LLAMA_FN_HOST_ACC_INLINE __host__ __device__ __forceinline__
		#    elif defined(__GNUC__) || defined(__clang__)
		#        define LLAMA_FN_HOST_ACC_INLINE inline __attribute__((always_inline))
		#    elif defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
		#        define LLAMA_FN_HOST_ACC_INLINE __forceinline
		#    else
		/// Some offloading parallelization language extensions such a CUDA, OpenACC or
		/// OpenMP 4.5 need to specify whether a class, struct, function or method
		/// "resides" on the host, the accelerator (the offloading device) or both.
		/// LLAMA supports this with marking every function needed on an accelerator
		/// with `LLAMA_FN_HOST_ACC_INLINE`. When using such a language (or e.g. <a
		/// href="https://github.com/alpaka-group/alpaka">alpaka</a>)
		/// this macro should be defined on the compiler's command line. E.g. for
		/// alpaka: -D'LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC'
		#        define LLAMA_FN_HOST_ACC_INLINE inline
		#        warning LLAMA_FN_HOST_ACC_INLINE not defined for this compiler
		#    endif
		#endif

		/// Suppresses nvcc warning: 'calling a __host__ function from __host__ __device__ function.'
		#if defined(__NVCC__) && !defined(__clang__)
		#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_exec_check_disable")
		#else
		#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING
		#endif

		#if defined(__INTEL_COMPILER) /*|| defined(__INTEL_LLVM_COMPILER)*/
		#    define LLAMA_FORCE_INLINE_RECURSIVE _Pragma("forceinline recursive")
		#elif defined(_MSC_VER)
		#    define LLAMA_FORCE_INLINE_RECURSIVE __pragma(inline_depth(255))
		#else
		/// Forces the compiler to recursively inline the call hiearchy started by the
		/// subsequent function call.
		#    define LLAMA_FORCE_INLINE_RECURSIVE
		#endif

		/// Forces a copy of a value. This is useful to prevent ODR usage of constants
		/// when compiling for GPU targets.
		#define LLAMA_COPY(x) decltype(x)(x)

		// TODO: clang 10 and 11 fail to compile this currently with the issue described here:
		// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
		// let's try again with clang 12
		// Intel LLVM compiler is also using the clang frontend
		#if (__has_include(<ranges>) && defined(__cpp_concepts) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER))
		#    define CAN_USE_RANGES 1
		#else
		#    define CAN_USE_RANGES 0
		#endif
		// ==
		// == ./macros.hpp ==
		// ============================================================================


	#include <tuple>

	namespace llama
	{
	    /// Array class like `std::array` but suitable for use with offloading
	    /// devices like GPUs.
	    /// \tparam T type if array elements.
	    /// \tparam N rank of the array.
	    template <typename T, std::size_t N>
	    struct Array
	    {
	        static constexpr std::size_t rank
	            = N; // FIXME this is right from the ArrayDims's POV, but wrong from the Array's POV
	        T element[N > 0 ? N : 1];

	        LLAMA_FN_HOST_ACC_INLINE constexpr T* begin()
	        {
	            return &element[0];
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr const T* begin() const
	        {
	            return &element[0];
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr T* end()
	        {
	            return &element[N];
	        };

	        LLAMA_FN_HOST_ACC_INLINE constexpr const T* end() const
	        {
	            return &element[N];
	        };

	        template <typename IndexType>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) -> T&
	        {
	            return element[idx];
	        }

	        template <typename IndexType>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) const -> T const&
	        {
	            return element[idx];
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator==(const Array& a, const Array& b) -> bool
	        {
	            for (std::size_t i = 0; i < N; ++i)
	                if (a.element[i] != b.element[i])
	                    return false;
	            return true;
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator!=(const Array& a, const Array& b) -> bool
	        {
	            return !(a == b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator+(const Array& a, const Array& b) -> Array
	        {
	            Array temp{};
	            for (std::size_t i = 0; i < N; ++i)
	                temp[i] = a[i] + b[i];
	            return temp;
	        }

	        template <std::size_t I>
	        constexpr auto get() -> T&
	        {
	            return element[I];
	        }

	        template <std::size_t I>
	        constexpr auto get() const -> const T&
	        {
	            return element[I];
	        }
	    };

	    template <typename First, typename... Args>
	    Array(First, Args... args) -> Array<First, sizeof...(Args) + 1>;
	} // namespace llama

	namespace std
	{
	    template <typename T, size_t N>
	    struct tuple_size<llama::Array<T, N>> : integral_constant<size_t, N>
	    {
	    };

	    template <size_t I, typename T, size_t N>
	    struct tuple_element<I, llama::Array<T, N>>
	    {
	        using type = T;
	    };
	} // namespace std
	// ==
	// == ./Array.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./Concepts.hpp ==
	// ==
	// #pragma once
	// #include "Array.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./Core.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Array.hpp"    // amalgamate: file already expanded
			// ============================================================================
			// == ./RecordCoord.hpp ==
			// ==
			// Copyright 2018 Alexander Matthes
			// SPDX-License-Identifier: GPL-3.0-or-later

			// #pragma once
			#include <array>
			#include <boost/mp11.hpp>
			#include <type_traits>

			namespace llama
			{
			    /// Represents a coordinate for a record inside the record dimension tree.
			    /// \tparam Coords... the compile time coordinate.
			    template <std::size_t... Coords>
			    struct RecordCoord
			    {
			        /// The list of integral coordinates as `boost::mp11::mp_list`.
			        using List = boost::mp11::mp_list_c<std::size_t, Coords...>;

			        static constexpr std::size_t front = boost::mp11::mp_front<List>::value;
			        static constexpr std::size_t back = boost::mp11::mp_back<List>::value;
			        static constexpr std::size_t size = sizeof...(Coords);
			    };

			    template <>
			    struct RecordCoord<>
			    {
			        using List = boost::mp11::mp_list_c<std::size_t>;

			        static constexpr std::size_t size = 0;
			    };

			    inline namespace literals
			    {
			        /// Literal operator for converting a numeric literal into a \ref RecordCoord.
			        template <char... Digits>
			        constexpr auto operator"" _RC()
			        {
			            constexpr auto coord = []() constexpr
			            {
			                char digits[] = {(Digits - 48)...};
			                std::size_t acc = 0;
			                std ::size_t powerOf10 = 1;
			                for (int i = sizeof...(Digits) - 1; i >= 0; i--)
			                {
			                    acc += digits[i] * powerOf10;
			                    powerOf10 *= 10;
			                }
			                return acc;
			            }
			            ();
			            return RecordCoord<coord>{};
			        }
			    } // namespace literals

			    namespace internal
			    {
			        template <class L>
			        struct mp_unwrap_sizes_impl;

			        template <template <class...> class L, typename... T>
			        struct mp_unwrap_sizes_impl<L<T...>>
			        {
			            using type = RecordCoord<T::value...>;
			        };

			        template <typename L>
			        using mp_unwrap_sizes = typename mp_unwrap_sizes_impl<L>::type;
			    } // namespace internal

			    /// Converts a type list of integral constants into a \ref RecordCoord.
			    template <typename L>
			    using RecordCoordFromList = internal::mp_unwrap_sizes<L>;

			    /// Concatenate two \ref RecordCoord.
			    template <typename RecordCoord1, typename RecordCoord2>
			    using Cat = RecordCoordFromList<boost::mp11::mp_append<typename RecordCoord1::List, typename RecordCoord2::List>>;

			    /// Concatenate two \ref RecordCoord instances.
			    template <typename RecordCoord1, typename RecordCoord2>
			    auto cat(RecordCoord1, RecordCoord2)
			    {
			        return Cat<RecordCoord1, RecordCoord2>{};
			    }

			    /// RecordCoord without first coordinate component.
			    template <typename RecordCoord>
			    using PopFront = RecordCoordFromList<boost::mp11::mp_pop_front<typename RecordCoord::List>>;

			    namespace internal
			    {
			        template <typename First, typename Second>
			        struct RecordCoordCommonPrefixIsBiggerImpl;

			        template <std::size_t... Coords1, std::size_t... Coords2>
			        struct RecordCoordCommonPrefixIsBiggerImpl<RecordCoord<Coords1...>, RecordCoord<Coords2...>>
			        {
			            static constexpr auto value = []() constexpr
			            {
			                // CTAD does not work if Coords1/2 is an empty pack
			                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
			                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
			                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
			                {
			                    if (a1[i] > a2[i])
			                        return true;
			                    if (a1[i] < a2[i])
			                        return false;
			                }
			                return false;
			            }
			            ();
			        };
			    } // namespace internal

			    /// Checks wether the first RecordCoord is bigger than the second.
			    template <typename First, typename Second>
			    inline constexpr auto RecordCoordCommonPrefixIsBigger
			        = internal::RecordCoordCommonPrefixIsBiggerImpl<First, Second>::value;

			    namespace internal
			    {
			        template <typename First, typename Second>
			        struct RecordCoordCommonPrefixIsSameImpl;

			        template <std::size_t... Coords1, std::size_t... Coords2>
			        struct RecordCoordCommonPrefixIsSameImpl<RecordCoord<Coords1...>, RecordCoord<Coords2...>>
			        {
			            static constexpr auto value = []() constexpr
			            {
			                // CTAD does not work if Coords1/2 is an empty pack
			                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
			                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
			                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
			                    if (a1[i] != a2[i])
			                        return false;
			                return true;
			            }
			            ();
			        };
			    } // namespace internal

			    /// Checks wether two \ref RecordCoord are the same or one is the prefix of
			    /// the other.
			    template <typename First, typename Second>
			    inline constexpr auto RecordCoordCommonPrefixIsSame
			        = internal::RecordCoordCommonPrefixIsSameImpl<First, Second>::value;
			} // namespace llama
			// ==
			// == ./RecordCoord.hpp ==
			// ============================================================================


		#include <boost/core/demangle.hpp>
		// #include <boost/mp11.hpp>    // amalgamate: file already included
		#include <iostream>
		// #include <type_traits>    // amalgamate: file already included

		namespace llama
		{
		    /// Anonymous naming for a \ref Field.
		    struct NoName
		    {
		    };

		    /// The run-time specified array dimensions.
		    /// \tparam Dim Compile-time number of dimensions.
		    template <std::size_t Dim>
		    struct ArrayDims : Array<std::size_t, Dim>
		    {
		    };

		    static_assert(std::is_trivially_default_constructible_v<ArrayDims<1>>); // so ArrayDims<1>{} will produce a zeroed
		                                                                            // coord. Should hold for all dimensions,
		                                                                            // but just checking for <1> here.

		    template <typename... Args>
		    ArrayDims(Args...) -> ArrayDims<sizeof...(Args)>;
		} // namespace llama

		template <size_t N>
		struct std::tuple_size<llama::ArrayDims<N>> : std::integral_constant<size_t, N>
		{
		};

		template <size_t I, size_t N>
		struct std::tuple_element<I, llama::ArrayDims<N>>
		{
		    using type = size_t;
		};

		namespace llama
		{
		    /// A type list of \ref Field which may be used to define a record dimension.
		    template <typename... Leaves>
		    struct Record
		    {
		    };

		    /// Record dimension tree node which may either be a leaf or refer to a child tree presented as another \ref
		    /// Record.
		    /// \tparam Tag Name of the node. May be any type (struct, class).
		    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting of a nested \ref
		    /// Record. 2. an array of any type, in which case a Record with as many \ref Field as the array
		    /// size is created, named \ref Index specialized on consecutive numbers. 3. A scalar type different from \ref
		    /// Record, making this node a leaf of this type.
		    template <typename Tag, typename Type>
		    struct Field
		    {
		    };

		    struct NrAndOffset
		    {
		        std::size_t nr;
		        std::size_t offset;

		        friend auto operator==(const NrAndOffset& a, const NrAndOffset& b) -> bool
		        {
		            return a.nr == b.nr && a.offset == b.offset;
		        }

		        friend auto operator!=(const NrAndOffset& a, const NrAndOffset& b) -> bool
		        {
		            return !(a == b);
		        }

		        friend auto operator<<(std::ostream& os, const NrAndOffset& value) -> std::ostream&
		        {
		            return os << "NrAndOffset{" << value.nr << ", " << value.offset << "}";
		        }
		    };

		    /// Get the tag from a \ref Field.
		    template <typename Field>
		    using GetFieldTag = boost::mp11::mp_first<Field>;

		    namespace internal
		    {
		        template <typename ChildType, std::size_t... Is>
		        auto makeRecordArray(std::index_sequence<Is...>)
		        {
		            return Record<Field<RecordCoord<Is>, ChildType>...>{};
		        }

		        template <typename T>
		        struct ArrayToRecord
		        {
		            using type = T;
		        };

		        template <typename ChildType, std::size_t Count>
		        struct ArrayToRecord<ChildType[Count]>
		        {
		            using type = decltype(internal::makeRecordArray<typename ArrayToRecord<ChildType>::type>(
		                std::make_index_sequence<Count>{}));
		        };
		    } // namespace internal

		    /// Get the type from a \ref Field.
		    template <typename Field>
		    using GetFieldType = typename internal::ArrayToRecord<boost::mp11::mp_second<Field>>::type;

		    template <typename T>
		    inline constexpr auto isRecord = false;

		    template <typename... Fields>
		    inline constexpr auto isRecord<Record<Fields...>> = true;

		    namespace internal
		    {
		        template <typename CurrTag, typename RecordDim, typename RecordCoord>
		        struct GetTagsImpl;

		        template <typename CurrTag, typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
		        struct GetTagsImpl<CurrTag, Record<Fields...>, RecordCoord<FirstCoord, Coords...>>
		        {
		            using Field = boost::mp11::mp_at_c<boost::mp11::mp_list<Fields...>, FirstCoord>;
		            using ChildTag = GetFieldTag<Field>;
		            using ChildType = GetFieldType<Field>;
		            using type = boost::mp11::
		                mp_push_front<typename GetTagsImpl<ChildTag, ChildType, RecordCoord<Coords...>>::type, CurrTag>;
		        };

		        template <typename CurrTag, typename T>
		        struct GetTagsImpl<CurrTag, T, RecordCoord<>>
		        {
		            using type = boost::mp11::mp_list<CurrTag>;
		        };
		    } // namespace internal

		    /// Get the tags of all \ref Field from the root of the record dimension
		    /// tree until to the node identified by \ref RecordCoord.
		    template <typename RecordDim, typename RecordCoord>
		    using GetTags = typename internal::GetTagsImpl<NoName, RecordDim, RecordCoord>::type;

		    /// Get the tag of the \ref Field at a \ref RecordCoord inside the
		    /// record dimension tree.
		    template <typename RecordDim, typename RecordCoord>
		    using GetTag = boost::mp11::mp_back<GetTags<RecordDim, RecordCoord>>;

		    /// Is true if, starting at two coordinates in two record dimensions, all
		    /// subsequent nodes in the record dimension tree have the same tag.
		    /// \tparam RecordDimA First record dimension.
		    /// \tparam LocalA \ref RecordCoord based on StartA along which the tags are
		    /// compared.
		    /// \tparam RecordDimB second record dimension.
		    /// \tparam LocalB \ref RecordCoord based on StartB along which the tags are
		    /// compared.
		    template <typename RecordDimA, typename LocalA, typename RecordDimB, typename LocalB>
		    inline constexpr auto hasSameTags = []() constexpr
		    {
		        if constexpr (LocalA::size != LocalB::size)
		            return false;
		        else if constexpr (LocalA::size == 0 && LocalB::size == 0)
		            return true;
		        else
		            return std::is_same_v<GetTags<RecordDimA, LocalA>, GetTags<RecordDimB, LocalB>>;
		    }
		    ();

		    namespace internal
		    {
		        template <typename RecordDim, typename RecordCoord, typename... Tags>
		        struct GetCoordFromTagsImpl
		        {
		            static_assert(boost::mp11::mp_size<RecordDim>::value != 0, "Tag combination is not valid");
		        };

		        template <typename... Fields, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
		        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<ResultCoords...>, FirstTag, Tags...>
		        {
		            template <typename Field>
		            struct HasTag : std::is_same<GetFieldTag<Field>, FirstTag>
		            {
		            };

		            static constexpr auto tagIndex = boost::mp11::mp_find_if<boost::mp11::mp_list<Fields...>, HasTag>::value;
		            static_assert(
		                tagIndex < sizeof...(Fields),
		                "FirstTag was not found inside this DatumStruct. Does your datum domain contain the tag you access "
		                "with?");

		            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Fields...>, tagIndex>>;

		            using type =
		                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., tagIndex>, Tags...>::type;
		        };

		        template <typename RecordDim, typename RecordCoord>
		        struct GetCoordFromTagsImpl<RecordDim, RecordCoord>
		        {
		            using type = RecordCoord;
		        };
		    } // namespace internal

		    /// Converts a series of tags navigating down a record dimension into a \ref RecordCoord.
		    template <typename RecordDim, typename... Tags>
		    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<RecordDim, RecordCoord<>, Tags...>::type;

		    namespace internal
		    {
		        template <typename RecordDim, typename... RecordCoordOrTags>
		        struct GetTypeImpl;

		        template <typename... Children, std::size_t HeadCoord, std::size_t... TailCoords>
		        struct GetTypeImpl<Record<Children...>, RecordCoord<HeadCoord, TailCoords...>>
		        {
		            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Children...>, HeadCoord>>;
		            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
		        };

		        template <typename T>
		        struct GetTypeImpl<T, RecordCoord<>>
		        {
		            using type = T;
		        };

		        template <typename RecordDim, typename... RecordCoordOrTags>
		        struct GetTypeImpl
		        {
		            using type = typename GetTypeImpl<RecordDim, GetCoordFromTags<RecordDim, RecordCoordOrTags...>>::type;
		        };
		    } // namespace internal

		    /// Returns the type of a node in a record dimension tree identified by a given
		    /// \ref RecordCoord or a series of tags.
		    template <typename RecordDim, typename... RecordCoordOrTags>
		    using GetType = typename internal::GetTypeImpl<RecordDim, RecordCoordOrTags...>::type;

		    namespace internal
		    {
		        template <typename RecordDim, typename BaseRecordCoord, typename... Tags>
		        struct GetCoordFromTagsRelativeImpl
		        {
		            using AbsolutCoord = typename internal::
		                GetCoordFromTagsImpl<GetType<RecordDim, BaseRecordCoord>, BaseRecordCoord, Tags...>::type;
		            // Only returning the record coord relative to BaseRecordCoord
		            using type
		                = RecordCoordFromList<boost::mp11::mp_drop_c<typename AbsolutCoord::List, BaseRecordCoord::size>>;
		        };
		    } // namespace internal

		    /// Converts a series of tags navigating down a record dimension, starting at a
		    /// given \ref RecordCoord, into a \ref RecordCoord.
		    template <typename RecordDim, typename BaseRecordCoord, typename... Tags>
		    using GetCoordFromTagsRelative =
		        typename internal::GetCoordFromTagsRelativeImpl<RecordDim, BaseRecordCoord, Tags...>::type;

		    namespace internal
		    {
		        template <typename T, std::size_t... Coords, typename Functor>
		        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafImpl(T*, RecordCoord<Coords...> coord, Functor&& functor)
		        {
		            functor(coord);
		        };

		        template <typename... Children, std::size_t... Coords, typename Functor>
		        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafImpl(
		            Record<Children...>*,
		            RecordCoord<Coords...>,
		            Functor&& functor)
		        {
		            LLAMA_FORCE_INLINE_RECURSIVE
		            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof...(Children)>>(
		                [&](auto i)
		                {
		                    constexpr auto childIndex = decltype(i)::value;
		                    using Field = boost::mp11::mp_at_c<Record<Children...>, childIndex>;

		                    LLAMA_FORCE_INLINE_RECURSIVE
		                    forEachLeafImpl(
		                        static_cast<GetFieldType<Field>*>(nullptr),
		                        RecordCoord<Coords..., childIndex>{},
		                        std::forward<Functor>(functor));
		                });
		        }
		    } // namespace internal

		    /// Iterates over the record dimension tree and calls a functor on each element.
		    /// \param functor Functor to execute at each element of. Needs to have
		    /// `operator()` with a template parameter for the \ref RecordCoord in the
		    /// record dimension tree.
		    /// \param baseCoord \ref RecordCoord at which the iteration should be
		    /// started. The functor is called on elements beneath this coordinate.
		    template <typename RecordDim, typename Functor, std::size_t... Coords>
		    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(Functor&& functor, RecordCoord<Coords...> baseCoord)
		    {
		        LLAMA_FORCE_INLINE_RECURSIVE
		        internal::forEachLeafImpl(
		            static_cast<GetType<RecordDim, RecordCoord<Coords...>>*>(nullptr),
		            baseCoord,
		            std::forward<Functor>(functor));
		    }

		    /// Iterates over the record dimension tree and calls a functor on each element.
		    /// \param functor Functor to execute at each element of. Needs to have
		    /// `operator()` with a template parameter for the \ref RecordCoord in the
		    /// record dimension tree.
		    /// \param baseTags Tags used to define where the iteration should be
		    /// started. The functor is called on elements beneath this coordinate.
		    template <typename RecordDim, typename Functor, typename... Tags>
		    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(Functor&& functor, Tags... baseTags)
		    {
		        LLAMA_FORCE_INLINE_RECURSIVE
		        forEachLeaf<RecordDim>(std::forward<Functor>(functor), GetCoordFromTags<RecordDim, Tags...>{});
		    }

		    namespace internal
		    {
		        template <typename T>
		        struct FlattenRecordDimImpl
		        {
		            using type = boost::mp11::mp_list<T>;
		        };

		        template <typename... Fields>
		        struct FlattenRecordDimImpl<Record<Fields...>>
		        {
		            using type = boost::mp11::mp_append<typename FlattenRecordDimImpl<GetFieldType<Fields>>::type...>;
		        };

		        // TODO: MSVC fails to compile if we move this function into an IILE at the callsite
		        template <typename RecordDim, typename RecordCoord>
		        constexpr auto flatRecordCoordImpl()
		        {
		            std::size_t c = 0;
		            forEachLeaf<RecordDim>(
		                [&](auto coord)
		                {
		                    if constexpr (RecordCoordCommonPrefixIsBigger<RecordCoord, decltype(coord)>)
		                        c++;
		                });
		            return c;
		        }
		    } // namespace internal

		    template <typename RecordDim>
		    using FlattenRecordDim = typename internal::FlattenRecordDimImpl<RecordDim>::type;

		    template <typename RecordDim, typename RecordCoord>
		    inline constexpr std::size_t flatRecordCoord = internal::flatRecordCoordImpl<RecordDim, RecordCoord>();

		    namespace internal
		    {
		        constexpr void roundUpToMultiple(std::size_t& value, std::size_t multiple)
		        {
		            value = ((value + multiple - 1) / multiple) * multiple;
		        }

		        // TODO: MSVC fails to compile if we move this function into an IILE at the callsite
		        template <bool Align, typename... Fields>
		        constexpr auto sizeOfRecordImpl()
		        {
		            using namespace boost::mp11;

		            std::size_t size = 0;
		            std::size_t maxAlign = 0;
		            using FlatRD = FlattenRecordDim<Record<Fields...>>;
		            mp_for_each<mp_transform<mp_identity, FlatRD>>([&](auto e) constexpr
		                                                           {
		                                                               using T = typename decltype(e)::type;
		                                                               if constexpr (Align)
		                                                                   roundUpToMultiple(size, alignof(T));
		                                                               maxAlign = std::max(maxAlign, alignof(T));
		                                                               size += sizeof(T);
		                                                           });

		            // final padding, so next struct can start right away
		            if constexpr (Align)
		                roundUpToMultiple(size, maxAlign);
		            return size;
		        }
		    } // namespace internal

		    template <typename T, bool Align = false>
		    inline constexpr std::size_t sizeOf = sizeof(T);

		    /// The size a record dimension if it would be a normal struct.
		    template <typename... Fields, bool Align>
		    inline constexpr std::size_t sizeOf<Record<Fields...>, Align> = internal::sizeOfRecordImpl<Align, Fields...>();

		    /// The byte offset of an element in a record dimension if it would be a normal struct.
		    /// \tparam RecordDim Record dimension tree.
		    /// \tparam RecordCoord Record coordinate of an element inrecord dimension tree.
		    template <typename RecordDim, typename RecordCoord, bool Align = false>
		    inline constexpr std::size_t offsetOf = []() constexpr
		    {
		        using namespace boost::mp11;

		        using FlatRD = FlattenRecordDim<RecordDim>;
		        constexpr auto flatCoord = flatRecordCoord<RecordDim, RecordCoord>;

		        std::size_t offset = 0;
		        mp_for_each<mp_iota_c<flatCoord>>([&](auto i) constexpr
		                                          {
		                                              using T = mp_at<FlatRD, decltype(i)>;
		                                              if constexpr (Align)
		                                                  internal::roundUpToMultiple(offset, alignof(T));
		                                              offset += sizeof(T);
		                                          });
		        if constexpr (Align)
		            internal::roundUpToMultiple(offset, alignof(mp_at_c<FlatRD, flatCoord>));
		        return offset;
		    }
		    ();


		    template <typename S>
		    auto structName(S) -> std::string
		    {
		        auto s = boost::core::demangle(typeid(S).name());
		        if (const auto pos = s.rfind(':'); pos != std::string::npos)
		            s = s.substr(pos + 1);
		        return s;
		    }

		    namespace internal
		    {
		        template <std::size_t Dim>
		        constexpr auto popFront(ArrayDims<Dim> ad)
		        {
		            ArrayDims<Dim - 1> result;
		            for (std::size_t i = 0; i < Dim - 1; i++)
		                result[i] = ad[i + 1];
		            return result;
		        }
		    } // namespace internal

		    template <std::size_t Dim, typename Func, typename... OuterIndices>
		    void forEachADCoord(ArrayDims<Dim> adSize, Func&& func, OuterIndices... outerIndices)
		    {
		        for (std::size_t i = 0; i < adSize[0]; i++)
		        {
		            if constexpr (Dim > 1)
		                forEachADCoord(internal::popFront(adSize), std::forward<Func>(func), outerIndices..., i);
		            else
		                std::forward<Func>(func)(ArrayDims<sizeof...(outerIndices) + 1>{outerIndices..., i});
		        }
		    }

		    namespace internal
		    {
		        template <typename T>
		        struct IndirectValue
		        {
		            T value;

		            auto operator->() -> T*
		            {
		                return &value;
		            }

		            auto operator->() const -> const T*
		            {
		                return &value;
		            }
		        };
		    } // namespace internal
		} // namespace llama
		// ==
		// == ./Core.hpp ==
		// ============================================================================


	// #include <type_traits>    // amalgamate: file already included

	#ifdef __cpp_concepts
	#    include <concepts>

	namespace llama
	{
	    // clang-format off
	    template <typename M>
	    concept Mapping = requires(M m) {
	        typename M::ArrayDims;
	        typename M::RecordDim;
	        { m.arrayDims() } -> std::same_as<typename M::ArrayDims>;
	        { M::blobCount } -> std::convertible_to<std::size_t>;
	        Array<int, M::blobCount>{}; // validates constexpr-ness
	        { m.blobSize(std::size_t{}) } -> std::same_as<std::size_t>;
	        { m.blobNrAndOffset(typename M::ArrayDims{}) } -> std::same_as<NrAndOffset>;
	    };
	    // clang-format on

	    template <typename B>
	    concept Blob = requires(B b, std::size_t i)
	    {
	        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can provide storage for
	        // other types
	        std::is_same_v<decltype(b[i]), std::byte&> || std::is_same_v<decltype(b[i]), unsigned char&>;
	    };

	    // clang-format off
	    template <typename BA>
	    concept BlobAllocator = requires(BA ba, std::size_t i) {
	        { ba(i) } -> Blob;
	    };
	    // clang-format on
	} // namespace llama

	#endif
	// ==
	// == ./Concepts.hpp ==
	// ============================================================================

// #include "macros.hpp"    // amalgamate: file already expanded

#include <cstddef>
#include <memory>
#include <vector>
#ifdef __INTEL_COMPILER
#    include <aligned_new>
#endif

namespace llama::bloballoc
{
    /// Allocates stack memory for a \ref View, which is copied each time a \ref
    /// View is copied.
    /// \tparam BytesToReserve the amount of memory to reserve.
    template <std::size_t BytesToReserve>
    struct Stack
    {
        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t) const -> Array<std::byte, BytesToReserve>
        {
            return {};
        }
    };
#ifdef __cpp_concepts
    static_assert(BlobAllocator<Stack<64>>);
#endif

    /// Allocates heap memory managed by a `std::shared_ptr` for a \ref View.
    /// This memory is shared between all copies of a \ref View.
    /// \tparam Alignment aligment of the allocated block of memory.
    template <std::size_t Alignment = 64>
    struct SharedPtr
    {
        inline auto operator()(std::size_t count) const -> std::shared_ptr<std::byte[]>
        {
            auto* ptr
                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
            auto deleter = [=](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
            return std::shared_ptr<std::byte[]>{ptr, deleter};
        }
    };
#ifdef __cpp_concepts
    static_assert(BlobAllocator<SharedPtr<>>);
#endif

    template <typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

        inline AlignedAllocator() noexcept = default;

        template <typename T2>
        inline AlignedAllocator(AlignedAllocator<T2, Alignment> const&) noexcept
        {
        }

        inline ~AlignedAllocator() noexcept = default;

        inline auto allocate(std::size_t n) -> T*
        {
            return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t{Alignment}));
        }

        inline void deallocate(T* p, std::size_t)
        {
            ::operator delete[](p, std::align_val_t{Alignment});
        }

        template <typename T2>
        struct rebind
        {
            using other = AlignedAllocator<T2, Alignment>;
        };

        auto operator!=(const AlignedAllocator<T, Alignment>& other) const -> bool
        {
            return !(*this == other);
        }

        auto operator==(const AlignedAllocator<T, Alignment>& other) const -> bool
        {
            return true;
        }
    };

    /// Allocates heap memory managed by a `std::vector` for a \ref View, which
    /// is copied each time a \ref View is copied.
    /// \tparam Alignment aligment of the allocated block of memory.
    template <std::size_t Alignment = 64u>
    struct Vector
    {
        inline auto operator()(std::size_t count) const
        {
            return std::vector<std::byte, AlignedAllocator<std::byte, Alignment>>(count);
        }
    };
#ifdef __cpp_concepts
    static_assert(BlobAllocator<Vector<>>);
#endif
} // namespace llama::bloballoc
// ==
// == ./BlobAllocators.hpp ==
// ============================================================================

// ============================================================================
// == ./DumpMapping.hpp ==
// ==
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
	// ============================================================================
	// == ./ArrayDimsIndexRange.hpp ==
	// ==
	// #pragma once
	// #include "Core.hpp"    // amalgamate: file already expanded

	#include <algorithm>
	#include <iterator>
	#if CAN_USE_RANGES
	#    include <ranges>
	#endif

	namespace llama
	{
	    /// Iterator supporting \ref ArrayDimsIndexRange.
	    template <std::size_t Dim>
	    struct ArrayDimsIndexIterator
	    {
	        using value_type = ArrayDims<Dim>;
	        using difference_type = std::ptrdiff_t;
	        using reference = value_type;
	        using pointer = internal::IndirectValue<value_type>;
	        using iterator_category = std::random_access_iterator_tag;

	        constexpr ArrayDimsIndexIterator() noexcept = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr ArrayDimsIndexIterator(ArrayDims<Dim> size, ArrayDims<Dim> current) noexcept
	            : lastIndex(
	                [size]() mutable
	                {
	                    for (auto i = 0; i < Dim; i++)
	                        size[i]--;
	                    return size;
	                }())
	            , current(current)
	        {
	        }

	        constexpr ArrayDimsIndexIterator(const ArrayDimsIndexIterator&) noexcept = default;
	        constexpr ArrayDimsIndexIterator(ArrayDimsIndexIterator&&) noexcept = default;
	        constexpr auto operator=(const ArrayDimsIndexIterator&) noexcept -> ArrayDimsIndexIterator& = default;
	        constexpr auto operator=(ArrayDimsIndexIterator&&) noexcept -> ArrayDimsIndexIterator& = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator*() const noexcept -> value_type
	        {
	            return current;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator->() const noexcept -> pointer
	        {
	            return {**this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++() noexcept -> ArrayDimsIndexIterator&
	        {
	            for (auto i = (int) Dim - 1; i >= 0; i--)
	            {
	                if (current[i] < lastIndex[i])
	                {
	                    current[i]++;
	                    return *this;
	                }
	                current[i] = 0;
	            }
	            current[0] = lastIndex[0] + 1;
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++(int) noexcept -> ArrayDimsIndexIterator
	        {
	            auto tmp = *this;
	            ++*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--() noexcept -> ArrayDimsIndexIterator&
	        {
	            for (auto i = (int) Dim - 1; i >= 0; i--)
	            {
	                if (current[i] > 0)
	                {
	                    current[i]--;
	                    return *this;
	                }
	                current[i] = lastIndex[i];
	            }
	            // decrementing beyond [0, 0, ..., 0] is UB
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--(int) noexcept -> ArrayDimsIndexIterator
	        {
	            auto tmp = *this;
	            --*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator[](difference_type i) const noexcept -> reference
	        {
	            return *(*this + i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator+=(difference_type n) noexcept -> ArrayDimsIndexIterator&
	        {
	            // add n to all lower dimensions with carry
	            for (auto i = (int) Dim - 1; i > 0 && n != 0; i--)
	            {
	                n += static_cast<difference_type>(current[i]);
	                const auto size = static_cast<difference_type>(lastIndex[i]) + 1;
	                auto mod = n % size;
	                n /= size;
	                if (mod < 0)
	                {
	                    mod += size;
	                    n--;
	                }
	                current[i] = mod;
	                assert(current[i] <= lastIndex[i]);
	            }

	            current[0] = static_cast<difference_type>(current[0]) + n;
	            // current is either within bounds or at the end ([last + 1, 0, 0, ..., 0])
	            assert(
	                (current[0] <= lastIndex[0]
	                 || (current[0] == lastIndex[0] + 1
	                     && std::all_of(std::begin(current) + 1, std::end(current), [](auto c) { return c == 0; })))
	                && "Iterator was moved past the end");

	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
	        {
	            it += n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(difference_type n, ArrayDimsIndexIterator it) noexcept -> ArrayDimsIndexIterator
	        {
	            return it + n;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator-=(difference_type n) noexcept -> ArrayDimsIndexIterator&
	        {
	            return operator+=(-n);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
	        {
	            it -= n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
	            -> difference_type
	        {
	            assert(a.lastIndex == b.lastIndex);

	            difference_type n = a.current[Dim - 1] - b.current[Dim - 1];
	            difference_type size = a.lastIndex[Dim - 1] + 1;
	            for (auto i = (int) Dim - 2; i >= 0; i--)
	            {
	                n += (a.current[i] - b.current[i]) * size;
	                size *= a.lastIndex[i] + 1;
	            }

	            return n;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator==(
	            const ArrayDimsIndexIterator<Dim>& a,
	            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
	        {
	            assert(a.lastIndex == b.lastIndex);
	            return a.current == b.current;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator!=(
	            const ArrayDimsIndexIterator<Dim>& a,
	            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
	        {
	            return !(a == b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
	            -> bool
	        {
	            assert(a.lastIndex == b.lastIndex);
	            return std::lexicographical_compare(
	                std::begin(a.current),
	                std::end(a.current),
	                std::begin(b.current),
	                std::end(b.current));
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
	            -> bool
	        {
	            return b < a;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
	            -> bool
	        {
	            return !(a > b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
	            -> bool
	        {
	            return !(a < b);
	        }

	    private:
	        ArrayDims<Dim> lastIndex;
	        ArrayDims<Dim> current;
	    };

	    /// Range allowing to iterate over all indices in a \ref ArrayDims.
	    template <std::size_t Dim>
	    struct ArrayDimsIndexRange
	#if CAN_USE_RANGES
	        : std::ranges::view_base
	#endif
	    {
	        constexpr ArrayDimsIndexRange() noexcept = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr ArrayDimsIndexRange(ArrayDims<Dim> size) noexcept : size(size)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto begin() const noexcept -> ArrayDimsIndexIterator<Dim>
	        {
	            return {size, ArrayDims<Dim>{}};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto end() const noexcept -> ArrayDimsIndexIterator<Dim>
	        {
	            auto endPos = ArrayDims<Dim>{};
	            endPos[0] = size[0];
	            return {size, endPos};
	        }

	    private:
	        ArrayDims<Dim> size;
	    };
	} // namespace llama
	// ==
	// == ./ArrayDimsIndexRange.hpp ==
	// ============================================================================

// #include "Core.hpp"    // amalgamate: file already expanded

#include <boost/container_hash/hash.hpp>
#include <fmt/format.h>
#include <string>
// #include <vector>    // amalgamate: file already included

namespace llama
{
    namespace internal
    {
        template <std::size_t... Coords>
        auto toVec(RecordCoord<Coords...>) -> std::vector<std::size_t>
        {
            return {Coords...};
        }

        template <typename Tag>
        auto tagToString(Tag tag)
        {
            return structName(tag);
        }

        // handle array indices
        template <std::size_t N>
        auto tagToString(RecordCoord<N>)
        {
            return std::to_string(N);
        }

        template <typename RecordDim, std::size_t... CoordsBefore, std::size_t CoordCurrent, std::size_t... CoordsAfter>
        void collectTagsAsStrings(
            std::vector<std::string>& v,
            RecordCoord<CoordsBefore...> before,
            RecordCoord<CoordCurrent, CoordsAfter...> after)
        {
            using Tag = GetTag<RecordDim, RecordCoord<CoordsBefore..., CoordCurrent>>;
            v.push_back(tagToString(Tag{}));
            if constexpr (sizeof...(CoordsAfter) > 0)
                collectTagsAsStrings<RecordDim>(
                    v,
                    RecordCoord<CoordsBefore..., CoordCurrent>{},
                    RecordCoord<CoordsAfter...>{});
        }

        template <typename RecordDim, std::size_t... Coords>
        auto tagsAsStrings(RecordCoord<Coords...>) -> std::vector<std::string>
        {
            std::vector<std::string> v;
            collectTagsAsStrings<RecordDim>(v, RecordCoord<>{}, RecordCoord<Coords...>{});
            return v;
        }

        template <typename Mapping, typename ArrayDims, std::size_t... Coords>
        auto mappingBlobNrAndOffset(const Mapping& mapping, const ArrayDims& adCoord, RecordCoord<Coords...>)
        {
            return mapping.template blobNrAndOffset<Coords...>(adCoord);
        }

        inline auto color(const std::vector<std::size_t>& recordCoord) -> std::size_t
        {
            auto c = (boost::hash_value(recordCoord) & 0xFFFFFF);
            const auto channelSum = ((c & 0xFF0000) >> 4) + ((c & 0xFF00) >> 2) + c & 0xFF;
            if (channelSum < 200)
                c |= 0x404040; // ensure color per channel is at least 0x40.
            return c;
        }

        template <std::size_t Dim>
        auto formatUdCoord(const ArrayDims<Dim>& coord)
        {
            if constexpr (Dim == 1)
                return std::to_string(coord[0]);
            else
            {
                std::string s = "{";
                for (auto v : coord)
                {
                    if (s.size() >= 2)
                        s += ",";
                    s += std::to_string(v);
                }
                s += "}";
                return s;
            }
        }

        inline auto formatDDTags(const std::vector<std::string>& tags)
        {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += ".";
                s += tag;
            }
            return s;
        }

        template <std::size_t Dim>
        struct FieldBox
        {
            ArrayDims<Dim> adCoord;
            std::vector<std::size_t> recordCoord;
            std::vector<std::string> recordTags;
            NrAndOffset nrAndOffset;
            std::size_t size;
        };

        template <typename Mapping>
        auto boxesFromMapping(const Mapping& mapping)
        {
            using ArrayDims = typename Mapping::ArrayDims;
            using RecordDim = typename Mapping::RecordDim;

            std::vector<FieldBox<Mapping::ArrayDims::rank>> infos;

            for (auto adCoord : ArrayDimsIndexRange{mapping.arrayDims()})
            {
                forEachLeaf<RecordDim>(
                    [&](auto coord)
                    {
                        constexpr int size = sizeof(GetType<RecordDim, decltype(coord)>);
                        infos.push_back(
                            {adCoord,
                             internal::toVec(coord),
                             internal::tagsAsStrings<RecordDim>(coord),
                             internal::mappingBlobNrAndOffset(mapping, adCoord, coord),
                             size});
                    });
            }

            return infos;
        }
    } // namespace internal

    /// Returns an SVG image visualizing the memory layout created by the given
    /// mapping. The created memory blocks are wrapped after wrapByteCount
    /// bytes.
    template <typename Mapping>
    auto toSvg(const Mapping& mapping, int wrapByteCount = 64) -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto blobBlockWidth = 60;

        const auto infos = internal::boxesFromMapping(mapping);

        std::string svg;
        svg += fmt::format(
            R"(<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg">
    <style>
        .label {{ font: {}px sans-serif; }}
    </style>
)",
            byteSizeInPixel / 2);

        std::array<int, Mapping::blobCount + 1> blobYOffset{};
        for (auto i = 0; i < Mapping::blobCount; i++)
        {
            const auto blobRows = (mapping.blobSize(i) + wrapByteCount - 1) / wrapByteCount;
            blobYOffset[i + 1] = blobYOffset[i] + (blobRows + 1) * byteSizeInPixel; // one row gap between blobs
            const auto height = blobRows * byteSizeInPixel;
            svg += fmt::format(
                R"a(<rect x="0" y="{}" width="{}" height="{}" fill="#AAA" stroke="#000"/>
<text x="{}" y="{}" fill="#000" text-anchor="middle">Blob: {}</text>
)a",
                blobYOffset[i],
                blobBlockWidth,
                height,
                blobBlockWidth / 2,
                blobYOffset[i] + height / 2,
                i);
        }

        for (const auto& info : infos)
        {
            const auto blobY = blobYOffset[info.nrAndOffset.nr];
            const auto x = (info.nrAndOffset.offset % wrapByteCount) * byteSizeInPixel + blobBlockWidth;
            const auto y = (info.nrAndOffset.offset / wrapByteCount) * byteSizeInPixel + blobY;

            const auto fill = internal::color(info.recordCoord);

            const auto width = byteSizeInPixel * info.size;
            svg += fmt::format(
                R"(<rect x="{}" y="{}" width="{}" height="{}" fill="#{:X}" stroke="#000"/>
)",
                x,
                y,
                width,
                byteSizeInPixel,
                fill);
            for (auto i = 1; i < info.size; i++)
            {
                svg += fmt::format(
                    R"(<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#777"/>
)",
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel * 2 / 3,
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel);
            }
            svg += fmt::format(
                R"(<text x="{}" y="{}" fill="#000" text-anchor="middle" class="label">{} {}</text>
)",
                x + width / 2,
                y + byteSizeInPixel * 3 / 4,
                internal::formatUdCoord(info.adCoord),
                internal::formatDDTags(info.recordTags));
        }
        svg += "</svg>";
        return svg;
    }

    /// Returns an HTML document visualizing the memory layout created by the
    /// given mapping. The visualization is resizeable.
    template <typename Mapping>
    auto toHtml(const Mapping& mapping) -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto rulerLengthInBytes = 512;
        constexpr auto rulerByteInterval = 8;

        auto infos = internal::boxesFromMapping(mapping);
        std::stable_sort(
            begin(infos),
            end(infos),
            [](const auto& a, const auto& b) {
                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
            });
        infos.erase(
            std::unique(
                begin(infos),
                end(infos),
                [](const auto& a, const auto& b) { return a.nrAndOffset == b.nrAndOffset; }),
            end(infos));

        auto cssClass = [](const std::vector<std::string>& tags)
        {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += "_";
                s += tag;
            }
            return s;
        };

        std::string svg;
        svg += fmt::format(
            R"(<!DOCTYPE html>
<html>
<head>
<style>
.box {{
    outline: 1px solid;
    display: inline-block;
    white-space: nowrap;
    height: {}px;
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #777 29px, #777 30px);
    text-align: center;
    overflow: hidden;
    vertical-align: middle;
}}
#ruler {{
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #000 29px, #000 30px);
    border-bottom: 1px solid;
    height: 20px;
    margin-bottom: 20px;
}}
#ruler div {{
    position: absolute;
    display: inline-block;
}}
)",
            byteSizeInPixel);
        using RecordDim = typename Mapping::RecordDim;
        forEachLeaf<RecordDim>(
            [&](auto coord)
            {
                constexpr int size = sizeof(GetType<RecordDim, decltype(coord)>);

                svg += fmt::format(
                    R"(.{} {{
    width: {}px;
    background-color: #{:X};
}}
)",
                    cssClass(internal::tagsAsStrings<RecordDim>(coord)),
                    byteSizeInPixel * size,
                    internal::color(internal::toVec(coord)));
            });

        svg += fmt::format(R"(</style>
</head>
<body>
    <header id="ruler">
)");
        for (auto i = 0; i < rulerLengthInBytes; i += rulerByteInterval)
            svg += fmt::format(
                R"(</style>
        <div style="margin-left: {}px;">{}</div>)",
                i * byteSizeInPixel,
                i);
        svg += fmt::format(R"(
    </header>
)");

        auto currentBlobNr = std::numeric_limits<std::size_t>::max();
        for (const auto& info : infos)
        {
            if (currentBlobNr != info.nrAndOffset.nr)
            {
                currentBlobNr = info.nrAndOffset.nr;
                svg += fmt::format("<h1>Blob: {}</h1>", currentBlobNr);
            }
            const auto width = byteSizeInPixel * info.size;
            svg += fmt::format(
                R"(<div class="box {0}" title="{1} {2}">{1} {2}</div>)",
                cssClass(info.recordTags),
                internal::formatUdCoord(info.adCoord),
                internal::formatDDTags(info.recordTags));
        }
        svg += R"(</body>
</html>)";
        return svg;
    }
} // namespace llama
// ==
// == ./DumpMapping.hpp ==
// ============================================================================

// ============================================================================
// == ./llama.hpp ==
// ==
// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
/// \mainpage LLAMA API documentation
///
/// LLAMA is a C++17 template header-only library for the abstraction of memory
/// access patterns. It distinguishes between the view of the algorithm on the
/// memory and the real layout in the background. This enables performance
/// portability for multicore, manycore and gpu applications with the very same
/// code.
///
/// In contrast to many other solutions LLAMA can define nested data structures
/// of arbitrary depths and is not limited only to struct of array and array of
/// struct data layouts. It is also capable to explicitly define padding,
/// blocking, striding and any other run time or compile time access pattern
/// simultaneously.
///
/// To archieve this goal LLAMA is split into mostly independent, orthogonal
/// parts completely written in modern C++17 to run on as many architectures and
/// with as many compilers as possible while still supporting extensions needed
/// e.g. to run on GPU or other many core hardware.
///
/// This page documents the API of LLAMA. The user documentation and an overview
/// about the concepts and ideas can be found here: https://llama-doc.rtfd.io
///
/// LLAMA is licensed under the LGPL3+.

#define LLAMA_VERSION_MAJOR 0
#define LLAMA_VERSION_MINOR 2
#define LLAMA_VERSION_PATCH 0

#ifdef __NVCC__
#    pragma push
#    pragma diag_suppress 940
#endif

// #include "ArrayDimsIndexRange.hpp"    // amalgamate: file already expanded
// #include "BlobAllocators.hpp"    // amalgamate: file already expanded
// #include "Core.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./View.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Array.hpp"    // amalgamate: file already expanded
	// #include "BlobAllocators.hpp"    // amalgamate: file already expanded
	// #include "Concepts.hpp"    // amalgamate: file already expanded
	// #include "Core.hpp"    // amalgamate: file already expanded
	// #include "macros.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/One.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already expanded

		namespace llama::mapping
		{
		    /// Maps all ArrayDims coordinates into the same location and layouts
		    /// struct members consecutively. This mapping is used for temporary, single
		    /// element views.
		    template <typename T_ArrayDims, typename T_RecordDim>
		    struct One
		    {
		        using ArrayDims = T_ArrayDims;
		        using RecordDim = T_RecordDim;

		        static constexpr std::size_t blobCount = 1;

		        constexpr One() = default;

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr One(ArrayDims, RecordDim = {})
		        {
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
		        {
		            // TODO: not sure if this is the right approach, since we take any ArrayDims in the ctor
		            ArrayDims ad;
		            for (auto i = 0; i < ArrayDims::rank; i++)
		                ad[i] = 1;
		            return ad;
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
		        {
		            return sizeOf<RecordDim>;
		        }

		        template <std::size_t... RecordCoords>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims) const -> NrAndOffset
		        {
		            constexpr auto offset = offsetOf<RecordDim, RecordCoord<RecordCoords...>>;
		            return {0, offset};
		        }
		    };
		} // namespace llama::mapping
		// ==
		// == ./mapping/One.hpp ==
		// ============================================================================


	// #include <type_traits>    // amalgamate: file already included

	namespace llama
	{
	#ifdef __cpp_concepts
	    template <typename T_Mapping, Blob BlobType>
	#else
	    template <typename T_Mapping, typename BlobType>
	#endif
	    struct View;

	    namespace internal
	    {
	        template <typename Allocator>
	        using AllocatorBlobType = decltype(std::declval<Allocator>()(0));

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template <typename Allocator, typename Mapping, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
	            const Allocator& alloc,
	            const Mapping& mapping,
	            std::integer_sequence<std::size_t, Is...>) -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
	        {
	            return {alloc(mapping.blobSize(Is))...};
	        }
	    } // namespace internal

	    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or \ref mapping::SoA. For allocating the
	    /// view's underlying memory, the specified allocator callable is used (or the default one, which is \ref
	    /// bloballoc::Vector). The allocator callable is called with the size of bytes to allocate for each blob of the
	    /// mapping. This function is the preferred way to create a \ref View.
	#ifdef __cpp_concepts
	    template <typename Mapping, BlobAllocator Allocator = bloballoc::Vector<>>
	#else
	    template <typename Mapping, typename Allocator = bloballoc::Vector<>>
	#endif
	    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
	        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
	    {
	        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
	        return {std::move(mapping), std::move(blobs)};
	    }

	    /// Allocates a \ref View holding a single record backed by stack memory (\ref bloballoc::Stack).
	    /// \tparam Dim Dimension of the \ref ArrayDims of the \ref View.
	    template <std::size_t Dim, typename RecordDim>
	    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
	    {
	        using Mapping = mapping::One<ArrayDims<Dim>, RecordDim>;
	        return allocView(Mapping{}, bloballoc::Stack<sizeOf<RecordDim>>{});
	    }

	    template <typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
	    struct VirtualRecord;

	    // TODO: Higher dimensional iterators might not have good codegen. Multiple nested loops seem to be superior to a
	    // single iterator over multiple dimensions. At least compilers are able to produce better code. std::mdspan also
	    // discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to remove/postpone iterators from
	    // the design. In std::mdspan's design, the iterator iterated over the co-domain.
	    template <typename View>
	    struct Iterator
	    {
	        using ADIterator = ArrayDimsIndexIterator<View::ArrayDims::rank>;

	        using iterator_category = std::random_access_iterator_tag;
	        using value_type = VirtualRecord<View>;
	        using difference_type = typename ADIterator::difference_type;
	        using pointer = internal::IndirectValue<value_type>;
	        using reference = value_type;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++() -> Iterator&
	        {
	            ++adIndex;
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++(int) -> Iterator
	        {
	            auto tmp = *this;
	            ++*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--() -> Iterator&
	        {
	            --adIndex;
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--(int) -> Iterator
	        {
	            auto tmp{*this};
	            --*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator*() const -> reference
	        {
	            return (*view)(*adIndex);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator->() const -> pointer
	        {
	            return {**this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator[](difference_type i) const -> reference
	        {
	            return *(*this + i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator+=(difference_type n) -> Iterator&
	        {
	            adIndex += n;
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(Iterator it, difference_type n) -> Iterator
	        {
	            it += n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(difference_type n, Iterator it) -> Iterator
	        {
	            return it + n;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator-=(difference_type n) -> Iterator&
	        {
	            adIndex -= n;
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(Iterator it, difference_type n) -> Iterator
	        {
	            it -= n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(const Iterator& a, const Iterator& b) -> difference_type
	        {
	            return static_cast<std::ptrdiff_t>(a.adIndex - b.adIndex);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
	        {
	            return a.adIndex == b.adIndex;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
	        {
	            return !(a == b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
	        {
	            return a.adIndex < b.adIndex;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>(const Iterator& a, const Iterator& b) -> bool
	        {
	            return b < a;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<=(const Iterator& a, const Iterator& b) -> bool
	        {
	            return !(a > b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>=(const Iterator& a, const Iterator& b) -> bool
	        {
	            return !(a < b);
	        }

	        ADIterator adIndex;
	        View* view;
	    };

	    namespace internal
	    {
	        template <typename Mapping, typename RecordCoord, typename = void>
	        struct isComputed : std::false_type
	        {
	        };

	        template <typename Mapping, typename RecordCoord>
	        struct isComputed<Mapping, RecordCoord, std::void_t<decltype(Mapping::isComputed(RecordCoord{}))>>
	            : std::bool_constant<Mapping::isComputed(RecordCoord{})>
	        {
	        };
	    } // namespace internal

	    /// Central LLAMA class holding memory for storage and giving access to
	    /// values stored there defined by a mapping. A view should be created using
	    /// \ref allocView.
	    /// \tparam T_Mapping The mapping used by the view to map accesses into
	    /// memory.
	    /// \tparam BlobType The storage type used by the view holding
	    /// memory.
	#ifdef __cpp_concepts
	    template <typename T_Mapping, Blob BlobType>
	#else
	    template <typename T_Mapping, typename BlobType>
	#endif
	    struct View
	    {
	        using Mapping = T_Mapping;
	        using ArrayDims = typename Mapping::ArrayDims;
	        using RecordDim = typename Mapping::RecordDim;
	        using VirtualRecordType = VirtualRecord<View>;
	        using VirtualRecordTypeConst = VirtualRecord<const View>;
	        using iterator = Iterator<View>;
	        using const_iterator = Iterator<const View>;

	        View() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        View(Mapping mapping, Array<BlobType, Mapping::blobCount> storageBlobs)
	            : mapping(std::move(mapping))
	            , storageBlobs(std::move(storageBlobs))
	        {
	        }

	        /// Retrieves the \ref VirtualRecord at the given \ref ArrayDims
	        /// coordinate.
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) const -> decltype(auto)
	        {
	            if constexpr (isRecord<RecordDim>)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return VirtualRecordTypeConst{arrayDims, *this};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return accessor(arrayDims, RecordCoord<>{});
	            }
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) -> decltype(auto)
	        {
	            if constexpr (isRecord<RecordDim>)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return VirtualRecordType{arrayDims, *this};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return accessor(arrayDims, RecordCoord<>{});
	            }
	        }

	        /// Retrieves the \ref VirtualRecord at the \ref ArrayDims coordinate
	        /// constructed from the passed component indices.
	        template <typename... Index>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) const -> decltype(auto)
	        {
	            static_assert(
	                sizeof...(Index) == ArrayDims::rank,
	                "Please specify as many indices as you have array dimensions");
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (ArrayDims{indices...});
	        }

	        template <typename... Index>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) -> decltype(auto)
	        {
	            static_assert(
	                sizeof...(Index) == ArrayDims::rank,
	                "Please specify as many indices as you have array dimensions");
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (ArrayDims{indices...});
	        }

	        /// Retrieves the \ref VirtualRecord at the \ref ArrayDims coordinate
	        /// constructed from the passed component indices.
	        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDims arrayDims) const -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (arrayDims);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDims arrayDims) -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (arrayDims);
	        }

	        /// Retrieves the \ref VirtualRecord at the 1D \ref ArrayDims coordinate
	        /// constructed from the passed index.
	        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) const -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (index);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this) (index);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto begin() -> iterator
	        {
	            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping.arrayDims()}.begin(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto begin() const -> const_iterator
	        {
	            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping.arrayDims()}.begin(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto end() -> iterator
	        {
	            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping.arrayDims()}.end(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto end() const -> const_iterator
	        {
	            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping.arrayDims()}.end(), this};
	        }

	        Mapping mapping;
	        Array<BlobType, Mapping::blobCount> storageBlobs;

	    private:
	        template <typename T_View, typename T_BoundRecordCoord, bool OwnView>
	        friend struct VirtualRecord;

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template <std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims, RecordCoord<Coords...> dc = {}) const
	            -> decltype(auto)
	        {
	            if constexpr (internal::isComputed<Mapping, RecordCoord<Coords...>>::value)
	                return mapping.compute(arrayDims, dc, storageBlobs);
	            else
	            {
	                const auto [nr, offset] = mapping.template blobNrAndOffset<Coords...>(arrayDims);
	                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
	                return reinterpret_cast<const Type&>(storageBlobs[nr][offset]);
	            }
	        }

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template <std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims, RecordCoord<Coords...> dc = {}) -> decltype(auto)
	        {
	            if constexpr (internal::isComputed<Mapping, RecordCoord<Coords...>>::value)
	                return mapping.compute(arrayDims, dc, storageBlobs);
	            else
	            {
	                const auto [nr, offset] = mapping.template blobNrAndOffset<Coords...>(arrayDims);
	                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
	                return reinterpret_cast<Type&>(storageBlobs[nr][offset]);
	            }
	        }
	    };

	    template <typename View>
	    inline constexpr auto IsView = false;

	    template <typename Mapping, typename BlobType>
	    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

	    /// Acts like a \ref View, but shows only a smaller and/or shifted part of
	    /// another view it references, the parent view.
	    template <typename T_ParentViewType>
	    struct VirtualView
	    {
	        using ParentView = T_ParentViewType; ///< type of the parent view
	        using Mapping = typename ParentView::Mapping; ///< mapping of the parent view
	        using ArrayDims = typename Mapping::ArrayDims; ///< array dimensions of the parent view
	        using VirtualRecordType = typename ParentView::VirtualRecordType; ///< VirtualRecord type of the
	                                                                          ///< parent view

	        /// Creates a VirtualView given a parent \ref View, offset and size.
	        LLAMA_FN_HOST_ACC_INLINE
	        VirtualView(ParentView& parentView, ArrayDims offset) : parentView(parentView), offset(offset)
	        {
	        }

	        template <std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims) const -> const auto&
	        {
	            return parentView.template accessor<Coords...>(ArrayDims{arrayDims + offset});
	        }

	        template <std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims) -> auto&
	        {
	            return parentView.template accessor<Coords...>(ArrayDims{arrayDims + offset});
	        }

	        /// Same as \ref View::operator()(ArrayDims), but shifted by the offset
	        /// of this \ref VirtualView.
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) const -> VirtualRecordType
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return parentView(ArrayDims{arrayDims + offset});
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) -> VirtualRecordType
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return parentView(ArrayDims{arrayDims + offset});
	        }

	        /// Same as corresponding operator in \ref View, but shifted by the
	        /// offset of this \ref VirtualView.
	        template <typename... Indices>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> VirtualRecordType
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return parentView(ArrayDims{ArrayDims{indices...} + offset});
	        }

	        template <typename... Indices>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> VirtualRecordType
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return parentView(ArrayDims{ArrayDims{indices...} + offset});
	        }

	        template <std::size_t... Coord>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>&& dc = {}) const -> const auto&
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return accessor<Coord...>(ArrayDims{});
	        }

	        template <std::size_t... Coord>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>&& dc = {}) -> auto&
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return accessor<Coord...>(ArrayDims{});
	        }

	        ParentView& parentView; ///< reference to parent view.
	        const ArrayDims offset; ///< offset this view's \ref ArrayDims coordinates are
	                                ///< shifted to the parent view.
	    };
	} // namespace llama
	// ==
	// == ./View.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./VirtualRecord.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "View.hpp"    // amalgamate: file already expanded

	// #include <type_traits>    // amalgamate: file already included

	namespace llama
	{
	    template <typename View, typename BoundRecordCoord, bool OwnView>
	    struct VirtualRecord;

	    template <typename View>
	    inline constexpr auto is_VirtualRecord = false;

	    template <typename View, typename BoundRecordCoord, bool OwnView>
	    inline constexpr auto is_VirtualRecord<VirtualRecord<View, BoundRecordCoord, OwnView>> = true;

	    /// A \ref VirtualRecord that owns and holds a single value.
	    template <typename RecordDim>
	    using One = VirtualRecord<decltype(allocViewStack<1, RecordDim>()), RecordCoord<>, true>;

	    /// Creates a single \ref VirtualRecord owning a view with stack memory and
	    /// copies all values from an existing \ref VirtualRecord.
	    template <typename VirtualRecord>
	    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualRecordStack(const VirtualRecord& vd) -> decltype(auto)
	    {
	        One<typename VirtualRecord::AccessibleRecordDim> temp;
	        temp = vd;
	        return temp;
	    }

	    namespace internal
	    {
	        template <
	            typename Functor,
	            typename LeftRecord,
	            typename RightView,
	            typename RightBoundRecordDim,
	            bool RightOwnView>
	        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(
	            LeftRecord& left,
	            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
	        {
	            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
	            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
	                [&](auto leftCoord)
	                {
	                    using LeftInnerCoord = decltype(leftCoord);
	                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
	                        [&](auto rightCoord)
	                        {
	                            using RightInnerCoord = decltype(rightCoord);
	                            if constexpr (hasSameTags<
	                                              typename LeftRecord::AccessibleRecordDim,
	                                              LeftInnerCoord,
	                                              typename RightRecord::AccessibleRecordDim,
	                                              RightInnerCoord>)
	                            {
	                                Functor{}(left(leftCoord), right(rightCoord));
	                            }
	                        });
	                });
	            return left;
	        }

	        template <typename Functor, typename LeftRecord, typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
	        {
	            forEachLeaf<typename LeftRecord::AccessibleRecordDim>([&](auto leftCoord)
	                                                                  { Functor{}(left(leftCoord), right); });
	            return left;
	        }

	        template <
	            typename Functor,
	            typename LeftRecord,
	            typename RightView,
	            typename RightBoundRecordDim,
	            bool RightOwnView>
	        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(
	            const LeftRecord& left,
	            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
	        {
	            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
	            bool result = true;
	            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
	                [&](auto leftCoord)
	                {
	                    using LeftInnerCoord = decltype(leftCoord);
	                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
	                        [&](auto rightCoord)
	                        {
	                            using RightInnerCoord = decltype(rightCoord);
	                            if constexpr (hasSameTags<
	                                              typename LeftRecord::AccessibleRecordDim,
	                                              LeftInnerCoord,
	                                              typename RightRecord::AccessibleRecordDim,
	                                              RightInnerCoord>)
	                            {
	                                result &= Functor{}(left(leftCoord), right(rightCoord));
	                            }
	                        });
	                });
	            return result;
	        }

	        template <typename Functor, typename LeftRecord, typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(const LeftRecord& left, const T& right) -> bool
	        {
	            bool result = true;
	            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
	                [&](auto leftCoord) {
	                    result &= Functor{}(
	                        left(leftCoord),
	                        static_cast<std::remove_reference_t<decltype(left(leftCoord))>>(right));
	                });
	            return result;
	        }

	        struct Assign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a = b;
	            }
	        };

	        struct PlusAssign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a += b;
	            }
	        };

	        struct MinusAssign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a -= b;
	            }
	        };

	        struct MultiplyAssign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a *= b;
	            }
	        };

	        struct DivideAssign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a /= b;
	            }
	        };

	        struct ModuloAssign
	        {
	            template <typename A, typename B>
	            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
	            {
	                return a %= b;
	            }
	        };

	        template <typename TWithOptionalConst, typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T) -> std::
	            enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::reference_wrapper<TWithOptionalConst>>
	        {
	            return leaf;
	        }

	        template <typename VirtualRecord, typename... Fields>
	        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualRecord&& vd, Record<Fields...>)
	        {
	            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
	        }

	        template <typename TWithOptionalConst, typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T)
	            -> std::enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::tuple<TWithOptionalConst&>>
	        {
	            return {leaf};
	        }

	        template <typename VirtualRecord, typename... Fields>
	        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualRecord&& vd, Record<Fields...>)
	        {
	            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
	        }

	        template <typename T, typename = void>
	        constexpr inline auto isTupleLike = false;

	        // get<I>(t) and std::tuple_size<T> must be available
	        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
	        template <typename T>
	        constexpr inline auto
	            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

	        template <typename... Ts>
	        constexpr inline auto dependentFalse = false;

	        template <typename Tuple1, typename Tuple2, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

	        template <typename T1, typename T2>
	        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
	        {
	            if constexpr (isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
	            {
	                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
	                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
	            }
	            else if constexpr (!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
	                std::forward<T1>(dst) = std::forward<T2>(src);
	            else
	                static_assert(dependentFalse<T1, T2>, "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
	        }

	        template <typename Tuple1, typename Tuple2, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
	        {
	            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
	            using std::get;
	            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
	        }

	        template <typename T, typename Tuple, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
	        {
	            using std::get;
	            return T{get<Is>(std::forward<Tuple>(src))...};
	        }

	        template <typename T, typename SFINAE, typename... Args>
	        constexpr inline auto isDirectListInitializableImpl = false;

	        template <typename T, typename... Args>
	        constexpr inline auto
	            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

	        template <typename T, typename... Args>
	        constexpr inline auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

	        template <typename T, typename Tuple>
	        constexpr inline auto isDirectListInitializableFromTuple = false;

	        template <typename T, template <typename...> typename Tuple, typename... Args>
	        constexpr inline auto
	            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;
	    } // namespace internal

	    /// Virtual record type returned by \ref View after resolving an array dimensions coordinate or partially resolving
	    /// a \ref RecordCoord. A virtual record does not hold data itself (thus named "virtual"), it just binds enough
	    /// information (array dimensions coord and partial record coord) to retrieve it from a \ref View later. Virtual
	    /// records should not be created by the user. They are returned from various access functions in \ref View and
	    /// VirtualRecord itself.
	    template <typename T_View, typename BoundRecordCoord, bool OwnView>
	    struct VirtualRecord
	    {
	        using View = T_View; ///< View this virtual record points into.

	    private:
	        using ArrayDims = typename View::Mapping::ArrayDims;
	        using RecordDim = typename View::Mapping::RecordDim;

	        const ArrayDims arrayDimsCoord;
	        std::conditional_t<OwnView, View, View&> view;

	    public:
	        /// Subtree of the record dimension of View starting at
	        /// BoundRecordCoord. If BoundRecordCoord is `RecordCoord<>` (default)
	        /// AccessibleRecordDim is the same as `Mapping::RecordDim`.
	        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

	        LLAMA_FN_HOST_ACC_INLINE VirtualRecord()
	            /* requires(OwnView) */
	            : arrayDimsCoord({})
	            , view{allocViewStack<1, RecordDim>()}
	        {
	            static_assert(OwnView, "The default constructor of VirtualRecord is only available if the ");
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        VirtualRecord(ArrayDims arrayDimsCoord, std::conditional_t<OwnView, View&&, View&> view)
	            : arrayDimsCoord(arrayDimsCoord)
	            , view{static_cast<decltype(view)>(view)}
	        {
	        }

	        VirtualRecord(const VirtualRecord&) = default;
	        VirtualRecord(VirtualRecord&&) = default;

	        /// Access a record in the record dimension underneath the current virtual
	        /// record using a \ref RecordCoord. If the access resolves to a leaf, a
	        /// reference to a variable inside the \ref View storage is returned,
	        /// otherwise another virtual record.
	        template <std::size_t... Coord>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
	        {
	            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
	            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return VirtualRecord<const View, AbsolutCoord>{arrayDimsCoord, this->view};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return this->view.accessor(arrayDimsCoord, AbsolutCoord{});
	            }
	        }

	        // FIXME(bgruber): remove redundancy
	        template <std::size_t... Coord>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> coord = {}) -> decltype(auto)
	        {
	            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
	            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return VirtualRecord<View, AbsolutCoord>{arrayDimsCoord, this->view};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return this->view.accessor(arrayDimsCoord, AbsolutCoord{});
	            }
	        }

	        /// Access a record in the record dimension underneath the current virtual
	        /// record using a series of tags. If the access resolves to a leaf, a
	        /// reference to a variable inside the \ref View storage is returned,
	        /// otherwise another virtual record.
	        template <typename... Tags>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
	        {
	            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

	            LLAMA_FORCE_INLINE_RECURSIVE
	            return operator()(RecordCoord{});
	        }

	        // FIXME(bgruber): remove redundancy
	        template <typename... Tags>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
	        {
	            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

	            LLAMA_FORCE_INLINE_RECURSIVE
	            return operator()(RecordCoord{});
	        }

	        // we need this one to disable the compiler generated copy assignment
	        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualRecord& other) -> VirtualRecord&
	        {
	            return this->operator=<VirtualRecord>(other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::Assign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::PlusAssign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::MinusAssign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::MultiplyAssign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::DivideAssign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> VirtualRecord&
	        {
	            return internal::virtualRecordArithOperator<internal::ModuloAssign>(*this, other);
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const VirtualRecord& vd, const T& t)
	        {
	            return copyVirtualRecordStack(vd) += t;
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const VirtualRecord& vd)
	        {
	            return vd + t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const VirtualRecord& vd, const T& t)
	        {
	            return copyVirtualRecordStack(vd) -= t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const VirtualRecord& vd, const T& t)
	        {
	            return copyVirtualRecordStack(vd) *= t;
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const VirtualRecord& vd)
	        {
	            return vd * t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const VirtualRecord& vd, const T& t)
	        {
	            return copyVirtualRecordStack(vd) /= t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const VirtualRecord& vd, const T& t)
	        {
	            return copyVirtualRecordStack(vd) %= t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::equal_to<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd == t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::not_equal_to<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd != t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::less<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd > t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::less_equal<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd >= t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::greater<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd < t;
	        }

	        template <typename T>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const VirtualRecord& vd, const T& t) -> bool
	        {
	            return internal::virtualRecordRelOperator<std::greater_equal<>>(vd, t);
	        }

	        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const VirtualRecord& vd) -> bool
	        {
	            return vd <= t;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
	        {
	            return internal::asTupleImpl(*this, AccessibleRecordDim{});
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
	        {
	            return internal::asTupleImpl(*this, AccessibleRecordDim{});
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
	        {
	            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
	        {
	            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
	        }

	        template <std::size_t I>
	        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
	        {
	            return operator()(RecordCoord<I>{});
	        }

	        template <std::size_t I>
	        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
	        {
	            return operator()(RecordCoord<I>{});
	        }

	        template <typename TupleLike>
	        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
	        {
	            static_assert(
	                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
	                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
	                "this: TupleLike{values...}");
	            return internal::makeFromTuple<TupleLike>(
	                asFlatTuple(),
	                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
	        }

	        template <typename TupleLike>
	        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
	        {
	            static_assert(
	                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
	                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
	                "this: TupleLike{values...}");
	            return internal::makeFromTuple<TupleLike>(
	                asFlatTuple(),
	                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
	        }

	        struct Loader
	        {
	            VirtualRecord& vd;

	            template <typename T>
	            LLAMA_FN_HOST_ACC_INLINE operator T()
	            {
	                return vd.loadAs<T>();
	            }
	        };

	        struct LoaderConst
	        {
	            const VirtualRecord& vd;

	            template <typename T>
	            LLAMA_FN_HOST_ACC_INLINE operator T() const
	            {
	                return vd.loadAs<T>();
	            }
	        };

	        LLAMA_FN_HOST_ACC_INLINE auto load() -> Loader
	        {
	            return {*this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto load() const -> LoaderConst
	        {
	            return {*this};
	        }

	        template <typename TupleLike>
	        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
	        {
	            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
	        }
	    };
	} // namespace llama

	template <typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_size<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
	    : boost::mp11::mp_size<typename llama::VirtualRecord<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
	{
	};

	template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_element<I, llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
	{
	    using type = decltype(std::declval<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
	};

	template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_element<I, const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
	{
	    using type
	        = decltype(std::declval<const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
	};
	// ==
	// == ./VirtualRecord.hpp ==
	// ============================================================================

// #include "macros.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/AoS.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
		// ============================================================================
		// == ./mapping/Common.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already expanded

		#include <climits>

		namespace llama::mapping
		{
		    namespace internal
		    {
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto product(const ArrayDims<Dim>& size) -> std::size_t
		        {
		            std::size_t prod = 1;
		            for (auto s : size)
		                prod *= s;
		            return prod;
		        }
		    } // namespace internal

		    /// Functor that maps a \ref ArrayDims coordinate into linear numbers the way C++ arrays work.
		    struct LinearizeArrayDimsCpp
		    {
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) -> std::size_t
		        {
		            return internal::product(size);
		        }

		        /// \param coord Coordinate in the array dimensions.
		        /// \param size Total size of the array dimensions.
		        /// \return Linearized index.
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDims<Dim>& coord, const ArrayDims<Dim>& size)
		            const -> std::size_t
		        {
		            std::size_t address = coord[0];
		            for (auto i = 1; i < Dim; i++)
		            {
		                address *= size[i];
		                address += coord[i];
		            }
		            return address;
		        }
		    };

		    /// Functor that maps a \ref ArrayDims coordinate into linear numbers the way Fortran arrays work.
		    struct LinearizeArrayDimsFortran
		    {
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) -> std::size_t
		        {
		            return internal::product(size);
		        }

		        /// \param coord Coordinate in the array dimensions.
		        /// \param size Total size of the array dimensions.
		        /// \return Linearized index.
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDims<Dim>& coord, const ArrayDims<Dim>& size)
		            const -> std::size_t
		        {
		            std::size_t address = coord[Dim - 1];
		            for (int i = (int) Dim - 2; i >= 0; i--)
		            {
		                address *= size[i];
		                address += coord[i];
		            }
		            return address;
		        }
		    };

		    /// Functor that maps a \ref ArrayDims coordinate into linear numbers using the Z-order space filling curve (Morton
		    /// codes).
		    struct LinearizeArrayDimsMorton
		    {
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) const -> std::size_t
		        {
		            std::size_t longest = size[0];
		            for (auto i = 1; i < Dim; i++)
		                longest = std::max(longest, size[i]);
		            const auto longestPO2 = bit_ceil(longest);
		            return intPow(longestPO2, Dim);
		        }

		        /// \param coord Coordinate in the array dimensions.
		        /// \param size Total size of the array dimensions.
		        /// \return Linearized index.
		        template <std::size_t Dim>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDims<Dim>& coord, const ArrayDims<Dim>&) const
		            -> std::size_t
		        {
		            std::size_t r = 0;
		            for (std::size_t bit = 0; bit < (sizeof(std::size_t) * CHAR_BIT) / Dim; bit++)
		                for (std::size_t i = 0; i < Dim; i++)
		                    r |= (coord[i] & (std::size_t{1} << bit)) << ((bit + 1) * (Dim - 1) - i);
		            return r;
		        }

		    private:
		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto bit_ceil(std::size_t n) -> std::size_t
		        {
		            std::size_t r = 1;
		            while (r < n)
		                r <<= 1;
		            return r;
		        }

		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto intPow(std::size_t b, std::size_t e) -> std::size_t
		        {
		            e--;
		            auto r = b;
		            while (e)
		            {
		                r *= b;
		                e--;
		            }
		            return r;
		        }
		    };
		} // namespace llama::mapping
		// ==
		// == ./mapping/Common.hpp ==
		// ============================================================================


	namespace llama::mapping
	{
	    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
	    /// If false, struct members are tighly packed.
	    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    template <
	        typename T_ArrayDims,
	        typename T_RecordDim,
	        bool AlignAndPad = false,
	        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct AoS
	    {
	        using ArrayDims = T_ArrayDims;
	        using RecordDim = T_RecordDim;
	        static constexpr std::size_t blobCount = 1;

	        constexpr AoS() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr AoS(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return arrayDimsSize;
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
	        {
	            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim, AlignAndPad>;
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
	                    * sizeOf<RecordDim, AlignAndPad> + offsetOf<RecordDim, RecordCoord<RecordCoords...>, AlignAndPad>;
	            return {0, offset};
	        }

	    private:
	        ArrayDims arrayDimsSize;
	    };

	    /// Array of struct mapping preserving the alignment of the element types by inserting padding. See \see AoS.
	    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using AlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

	    /// Array of struct mapping packing the element types tighly, violating the types alignment requirements. See \see
	    /// AoS.
	    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using PackedAoS = AoS<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

	    template <bool AlignAndPad = false, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct PreconfiguredAoS
	    {
	        template <typename ArrayDims, typename RecordDim>
	        using type = AoS<ArrayDims, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor>;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/AoS.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/AoSoA.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	#include <limits>

	namespace llama::mapping
	{
	    /// The maximum number of vector lanes that can be used to fetch each leaf type in the record dimension into a
	    /// vector register of the given size in bits.
	    template <typename RecordDim, std::size_t VectorRegisterBits>
	    inline constexpr std::size_t maxLanes = []() constexpr
	    {
	        auto max = std::numeric_limits<std::size_t>::max();
	        forEachLeaf<RecordDim>(
	            [&](auto coord)
	            {
	                using AttributeType = GetType<RecordDim, decltype(coord)>;
	                max = std::min(max, VectorRegisterBits / (sizeof(AttributeType) * CHAR_BIT));
	            });
	        return max;
	    }
	    ();

	    /// Array of struct of arrays mapping. Used to create a \ref View via \ref
	    /// allocView.
	    /// \tparam Lanes The size of the inner arrays of this array of struct of
	    /// arrays.
	    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    template <
	        typename T_ArrayDims,
	        typename T_RecordDim,
	        std::size_t Lanes,
	        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct AoSoA
	    {
	        using ArrayDims = T_ArrayDims;
	        using RecordDim = T_RecordDim;
	        static constexpr std::size_t blobCount = 1;

	        constexpr AoSoA() = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr AoSoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return arrayDimsSize;
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
	        {
	            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            const auto flatArrayIndex = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize);
	            const auto blockIndex = flatArrayIndex / Lanes;
	            const auto laneIndex = flatArrayIndex % Lanes;
	            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex
	                + offsetOf<RecordDim, RecordCoord<RecordCoords...>> * Lanes
	                + sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>) * laneIndex;
	            return {0, offset};
	        }

	    private:
	        ArrayDims arrayDimsSize;
	    };

	    template <std::size_t Lanes, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct PreconfiguredAoSoA
	    {
	        template <typename ArrayDims, typename RecordDim>
	        using type = AoSoA<ArrayDims, RecordDim, Lanes, LinearizeArrayDimsFunctor>;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/AoSoA.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Heatmap.hpp ==
	// ==
	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <array>    // amalgamate: file already included
	#include <atomic>
	#include <sstream>
	// #include <vector>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// Forwards all calls to the inner mapping. Counts all accesses made to all bytes, allowing to extract a heatmap.
	    /// \tparam Mapping The type of the inner mapping.
	    template <typename Mapping, typename CountType = std::size_t>
	    struct Heatmap
	    {
	        using ArrayDims = typename Mapping::ArrayDims;
	        using RecordDim = typename Mapping::RecordDim;
	        static constexpr std::size_t blobCount = Mapping::blobCount;

	        constexpr Heatmap() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        Heatmap(Mapping mapping) : mapping(mapping)
	        {
	            for (auto i = 0; i < blobCount; i++)
	                byteHits[i] = std::vector<std::atomic<CountType>>(blobSize(i));
	        }

	        Heatmap(const Heatmap&) = delete;
	        auto operator=(const Heatmap&) -> Heatmap& = delete;

	        Heatmap(Heatmap&&) noexcept = default;
	        auto operator=(Heatmap&&) noexcept -> Heatmap& = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return mapping.arrayDims();
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return mapping.blobSize(i);
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            const auto nao = mapping.template blobNrAndOffset<RecordCoords...>(coord);
	            for (auto i = 0; i < sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>); i++)
	                byteHits[nao.nr][nao.offset + i]++;
	            return nao;
	        }

	        // gnuplot with:
	        // set view map
	        // set yrange [] reverse
	        // splot "./file.dat" matrix with image
	        auto toGnuplotDatFile() const -> std::string
	        {
	            std::stringstream f;
	            for (auto i = 0; i < blobCount; i++)
	            {
	                std::size_t byteCount = 0;
	                for (const auto& hits : byteHits[i])
	                    f << hits << ((++byteCount % 64 == 0) ? '\n' : ' ');
	                while (byteCount++ % 64 != 0)
	                    f << "0 ";
	                f << '\n';
	            }
	            return f.str();
	        }

	        Mapping mapping;
	        mutable std::array<std::vector<std::atomic<CountType>>, blobCount> byteHits;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/Heatmap.hpp ==
	// ============================================================================

// #include "mapping/One.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/SoA.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <limits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
	    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    template <
	        typename T_ArrayDims,
	        typename T_RecordDim,
	        bool SeparateBuffers = false,
	        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct SoA
	    {
	        using ArrayDims = T_ArrayDims;
	        using RecordDim = T_RecordDim;
	        static constexpr std::size_t blobCount
	            = SeparateBuffers ? boost::mp11::mp_size<FlattenRecordDim<RecordDim>>::value : 1;

	        constexpr SoA() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr SoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return arrayDimsSize;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(std::size_t blobIndex) const -> std::size_t
	        {
	            if constexpr (SeparateBuffers)
	            {
	                constexpr Array<std::size_t, blobCount> typeSizes = []() constexpr
	                {
	                    Array<std::size_t, blobCount> r{};
	                    forEachLeaf<RecordDim>([&r, i = 0](auto coord) mutable constexpr
	                                           { r[i++] = sizeof(GetType<RecordDim, decltype(coord)>); });
	                    return r;
	                }
	                ();
	                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * typeSizes[blobIndex];
	            }
	            else
	                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            if constexpr (SeparateBuffers)
	            {
	                constexpr auto blob = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
	                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
	                    * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>);
	                return {blob, offset};
	            }
	            else
	            {
	                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
	                        * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>)
	                    + offsetOf<
	                          RecordDim,
	                          RecordCoord<RecordCoords...>> * LinearizeArrayDimsFunctor{}.size(arrayDimsSize);
	                return {0, offset};
	            }
	        }

	    private:
	        ArrayDims arrayDimsSize;
	    };

	    /// Struct of array mapping storing the entire layout in a single blob. See \see SoA.
	    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using SingleBlobSoA = SoA<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

	    /// Struct of array mapping storing each attribute of the record dimension in a separate blob. See \see SoA.
	    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using MultiBlobSoA = SoA<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

	    template <bool SeparateBuffers = false, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct PreconfiguredSoA
	    {
	        template <typename ArrayDims, typename RecordDim>
	        using type = SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/SoA.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Split.hpp ==
	// ==
	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template <typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
	        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
	        {
	            using namespace boost::mp11;
	            if constexpr (sizeof...(Coords) == 0)
	            {
	                using With = Record<mp_at_c<Record<Fields...>, FirstCoord>>;
	                using Without = mp_erase_c<Record<Fields...>, FirstCoord, FirstCoord + 1>;
	                return mp_list<With, Without>{};
	            }
	            else
	            {
	                using Result = decltype(partitionRecordDim(
	                    Record<mp_at_c<Record<Fields...>, FirstCoord>>{},
	                    RecordCoord<Coords...>{}));
	                using With = mp_replace_at_c<Record<Fields...>, FirstCoord, mp_first<Result>>;
	                using Without = mp_replace_at_c<Record<Fields...>, FirstCoord, mp_second<Result>>;
	                return mp_list<With, Without>{};
	            }
	        }

	        template <
	            std::size_t FirstDstCoord,
	            std::size_t... DstCoords,
	            std::size_t FirstSkippedCoord,
	            std::size_t... SkippedCoords>
	        constexpr auto offsetCoord(
	            RecordCoord<FirstDstCoord, DstCoords...>,
	            RecordCoord<FirstSkippedCoord, SkippedCoords...>)
	        {
	            if constexpr (FirstDstCoord < FirstSkippedCoord)
	                return RecordCoord<FirstDstCoord, DstCoords...>{};
	            else if constexpr (FirstDstCoord > FirstSkippedCoord)
	                return RecordCoord<FirstDstCoord - 1, DstCoords...>{};
	            else
	                return cat(
	                    RecordCoord<FirstDstCoord>{},
	                    offsetCoord(RecordCoord<DstCoords...>{}, RecordCoord<SkippedCoords...>{}));
	        }
	    } // namespace internal

	    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
	    /// \tparam RecordCoordForMapping1 A \ref RecordCoord selecting the part of the record dimension to be mapped
	    /// differently. \tparam MappingTemplate1 The mapping used for the selected part of the record dimension. \tparam
	    /// MappingTemplate2 The mapping used for the not selected part of the record dimension. \tparam SeparateBlobs If
	    /// true, both pieces of the record dimension are mapped to separate blobs.
	    template <
	        typename T_ArrayDims,
	        typename T_RecordDim,
	        typename RecordCoordForMapping1,
	        template <typename...>
	        typename MappingTemplate1,
	        template <typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct Split
	    {
	        using ArrayDims = T_ArrayDims;
	        using RecordDim = T_RecordDim;

	        using RecordDimPartitions = decltype(internal::partitionRecordDim(RecordDim{}, RecordCoordForMapping1{}));
	        using RecordDim1 = boost::mp11::mp_first<RecordDimPartitions>;
	        using RecordDim2 = boost::mp11::mp_second<RecordDimPartitions>;

	        using Mapping1 = MappingTemplate1<ArrayDims, RecordDim1>;
	        using Mapping2 = MappingTemplate2<ArrayDims, RecordDim2>;

	        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
	        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
	        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

	        constexpr Split() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr Split(ArrayDims size) : mapping1(size), mapping2(size)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return mapping1.arrayDims();
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
	        {
	            if constexpr (SeparateBlobs)
	            {
	                if (i < Mapping1::blobCount)
	                    return mapping1.blobSize(i);
	                else
	                    return mapping2.blobSize(i - Mapping1::blobCount);
	            }
	            else
	                return mapping1.blobSize(0) + mapping2.blobSize(0);
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            if constexpr (RecordCoordCommonPrefixIsSame<RecordCoordForMapping1, RecordCoord<RecordCoords...>>)
	            {
	                using namespace boost::mp11;
	                // zero all coordinate values that are part of RecordCoordForMapping1
	                using Prefix = mp_repeat_c<mp_list_c<std::size_t, 0>, RecordCoordForMapping1::size>;
	                using Suffix = mp_drop_c<mp_list_c<std::size_t, RecordCoords...>, RecordCoordForMapping1::size>;
	                return blobNrAndOffset(RecordCoordFromList<mp_append<Prefix, Suffix>>{}, coord, mapping1);
	            }
	            else
	            {
	                constexpr auto dstCoord
	                    = internal::offsetCoord(RecordCoord<RecordCoords...>{}, RecordCoordForMapping1{});
	                auto nrAndOffset = blobNrAndOffset(dstCoord, coord, mapping2);
	                if constexpr (SeparateBlobs)
	                    nrAndOffset.nr += Mapping1::blobCount;
	                else
	                {
	                    for (auto i = 0; i < Mapping1::blobCount; i++)
	                        nrAndOffset.offset += mapping1.blobSize(i);
	                }
	                return nrAndOffset;
	            }
	        }

	    private:
	        template <std::size_t... RecordCoords, typename Mapping>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            RecordCoord<RecordCoords...>,
	            ArrayDims coord,
	            const Mapping& mapping) const -> NrAndOffset
	        {
	            return mapping.template blobNrAndOffset<RecordCoords...>(coord);
	        }

	    public:
	        Mapping1 mapping1;
	        Mapping2 mapping2;
	    };

	    template <
	        typename RecordCoordForMapping1,
	        template <typename...>
	        typename MappingTemplate1,
	        template <typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct PreconfiguredSplit
	    {
	        template <typename ArrayDims, typename RecordDim>
	        using type
	            = Split<ArrayDims, RecordDim, RecordCoordForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/Split.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Trace.hpp ==
	// ==
	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <atomic>    // amalgamate: file already included
	// #include <boost/core/demangle.hpp>    // amalgamate: file already included
	// #include <iostream>    // amalgamate: file already included
	// #include <string>    // amalgamate: file already included
	#include <unordered_map>
	// #include <vector>    // amalgamate: file already included

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template <typename RecordDim, std::size_t... Coords>
	        auto coordName(RecordCoord<Coords...>) -> std::string
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<Coords...>>;

	            std::string r;
	            boost::mp11::mp_for_each<Tags>(
	                [&](auto tag)
	                {
	                    if (!r.empty())
	                        r += '.';
	                    r += structName(tag);
	                });
	            return r;
	        }
	    } // namespace internal

	    /// Forwards all calls to the inner mapping. Traces all accesses made
	    /// through this mapping and prints a summary on destruction.
	    /// \tparam Mapping The type of the inner mapping.
	    template <typename Mapping>
	    struct Trace
	    {
	        using ArrayDims = typename Mapping::ArrayDims;
	        using RecordDim = typename Mapping::RecordDim;
	        static constexpr std::size_t blobCount = Mapping::blobCount;

	        constexpr Trace() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        Trace(Mapping mapping) : mapping(mapping)
	        {
	            forEachLeaf<RecordDim>([&](auto coord) { fieldHits[internal::coordName<RecordDim>(coord)] = 0; });
	        }

	        Trace(const Trace&) = delete;
	        auto operator=(const Trace&) -> Trace& = delete;

	        Trace(Trace&&) noexcept = default;
	        auto operator=(Trace&&) noexcept -> Trace& = default;

	        ~Trace()
	        {
	            if (!fieldHits.empty())
	            {
	                std::cout << "Trace mapping, number of accesses:\n";
	                for (const auto& [k, v] : fieldHits)
	                    std::cout << '\t' << k << ":\t" << v << '\n';
	            }
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
	        {
	            return mapping.arrayDims();
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return mapping.blobSize(i);
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            const static auto name = internal::coordName<RecordDim>(RecordCoord<RecordCoords...>{});
	            fieldHits.at(name)++;

	            LLAMA_FORCE_INLINE_RECURSIVE return mapping.template blobNrAndOffset<RecordCoords...>(coord);
	        }

	        Mapping mapping;
	        mutable std::unordered_map<std::string, std::atomic<std::size_t>> fieldHits;
	    };
	} // namespace llama::mapping
	// ==
	// == ./mapping/Trace.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/tree/Mapping.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../Common.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/tree/Functors.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
			// ============================================================================
			// == ./mapping/tree/TreeFromDimensions.hpp ==
			// ==
			// Copyright 2018 Alexander Matthes
			// SPDX-License-Identifier: GPL-3.0-or-later
			// #pragma once
			// #include "../../Core.hpp"    // amalgamate: file already expanded
				// ============================================================================
				// == ./Tuple.hpp ==
				// ==
				// Copyright 2018 Alexander Matthes
				// SPDX-License-Identifier: GPL-3.0-or-later

				// #pragma once
				// #include "macros.hpp"    // amalgamate: file already expanded

				// #include <boost/mp11.hpp>    // amalgamate: file already included

				namespace llama
				{
				    /// Tuple class like `std::tuple` but suitable for use with offloading
				    /// devices like GPUs.
				    template <typename... Elements>
				    struct Tuple;

				    template <>
				    struct Tuple<>
				    {
				    };

				    template <typename T_FirstElement, typename... Elements>
				    struct Tuple<T_FirstElement, Elements...>
				    {
				        using FirstElement = T_FirstElement;
				        using RestTuple = Tuple<Elements...>;

				        constexpr Tuple() = default;

				        LLAMA_FN_HOST_ACC_INLINE
				        constexpr Tuple(FirstElement first, Elements... rest) : first(first), rest(rest...)
				        {
				        }

				        LLAMA_FN_HOST_ACC_INLINE
				        constexpr Tuple(FirstElement first, Tuple<Elements...> rest) : first(first), rest(rest)
				        {
				        }

				        FirstElement first; ///< the first element (if existing)
				        RestTuple rest; ///< the remaining elements
				    };

				    template <typename T_FirstElement>
				    struct Tuple<T_FirstElement>
				    {
				        using FirstElement = T_FirstElement;
				        using RestTuple = Tuple<>;

				        constexpr Tuple() = default;

				        LLAMA_FN_HOST_ACC_INLINE
				        constexpr Tuple(FirstElement const first, Tuple<> const rest = Tuple<>()) : first(first)
				        {
				        }

				        FirstElement first;
				    };

				    template <typename... Elements>
				    Tuple(Elements...) -> Tuple<Elements...>;

				    template <typename Tuple, std::size_t Pos>
				    using TupleElement = boost::mp11::mp_at_c<Tuple, Pos>;

				    template <std::size_t Pos, typename... Elements>
				    LLAMA_FN_HOST_ACC_INLINE auto get(const Tuple<Elements...>& tuple)
				    {
				        if constexpr (Pos == 0)
				            return tuple.first;
				        else
				            return get<Pos - 1>(tuple.rest);
				    }

				    template <typename Tuple>
				    inline constexpr auto tupleSize = boost::mp11::mp_size<Tuple>::value;

				    namespace internal
				    {
				        template <typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
				        static LLAMA_FN_HOST_ACC_INLINE auto tupleCatImpl(
				            const Tuple1& t1,
				            const Tuple2& t2,
				            std::index_sequence<Is1...>,
				            std::index_sequence<Is2...>)
				        {
				            return Tuple{get<Is1>(t1)..., get<Is2>(t2)...};
				        }
				    } // namespace internal

				    template <typename Tuple1, typename Tuple2>
				    LLAMA_FN_HOST_ACC_INLINE auto tupleCat(const Tuple1& t1, const Tuple2& t2)
				    {
				        return internal::tupleCatImpl(
				            t1,
				            t2,
				            std::make_index_sequence<tupleSize<Tuple1>>{},
				            std::make_index_sequence<tupleSize<Tuple2>>{});
				    }

				    namespace internal
				    {
				        template <std::size_t Pos, typename Tuple, typename Replacement>
				        struct TupleReplaceImpl
				        {
				            LLAMA_FN_HOST_ACC_INLINE
				            auto operator()(Tuple const tuple, Replacement const replacement)
				            {
				                return tupleCat(
				                    llama::Tuple{tuple.first},
				                    TupleReplaceImpl<Pos - 1, typename Tuple::RestTuple, Replacement>()(tuple.rest, replacement));
				            };
				        };

				        template <typename... Elements, typename Replacement>
				        struct TupleReplaceImpl<0, Tuple<Elements...>, Replacement>
				        {
				            LLAMA_FN_HOST_ACC_INLINE
				            auto operator()(Tuple<Elements...> tuple, Replacement const replacement)
				            {
				                return tupleCat(Tuple{replacement}, tuple.rest);
				            };
				        };

				        template <typename OneElement, typename Replacement>
				        struct TupleReplaceImpl<0, Tuple<OneElement>, Replacement>
				        {
				            LLAMA_FN_HOST_ACC_INLINE
				            auto operator()(Tuple<OneElement>, Replacement const replacement)
				            {
				                return Tuple{replacement};
				            }
				        };
				    } // namespace internal

				    /// Creates a copy of a tuple with the element at position Pos replaced by
				    /// replacement.
				    template <std::size_t Pos, typename Tuple, typename Replacement>
				    LLAMA_FN_HOST_ACC_INLINE auto tupleReplace(Tuple tuple, Replacement replacement)
				    {
				        return internal::TupleReplaceImpl<Pos, Tuple, Replacement>()(tuple, replacement);
				    }

				    namespace internal
				    {
				        template <typename Seq>
				        struct TupleTransformHelper;

				        template <size_t... Is>
				        struct TupleTransformHelper<std::index_sequence<Is...>>
				        {
				            template <typename... Elements, typename Functor>
				            static LLAMA_FN_HOST_ACC_INLINE auto transform(const Tuple<Elements...>& tuple, const Functor& functor)
				            {
				                // FIXME(bgruber): nvcc fails to compile
				                // Tuple{functor(get<Is>(tuple))...}
				                return Tuple<decltype(functor(std::declval<Elements>()))...>{functor(get<Is>(tuple))...};
				            }
				        };
				    } // namespace internal

				    /// Applies a functor to every element of a tuple, creating a new tuple with
				    /// the result of the element transformations. The functor needs to
				    /// implement a template `operator()` to which all tuple elements are
				    /// passed.
				    // TODO: replace by mp11 version in Boost 1.74.
				    template <typename... Elements, typename Functor>
				    LLAMA_FN_HOST_ACC_INLINE auto tupleTransform(const Tuple<Elements...>& tuple, const Functor& functor)
				    {
				        return internal::TupleTransformHelper<std::make_index_sequence<sizeof...(Elements)>>::transform(tuple, functor);
				    }

				    /// Returns a copy of the tuple without the first element.
				    template <typename... Elements>
				    LLAMA_FN_HOST_ACC_INLINE auto tupleWithoutFirst(const Tuple<Elements...>& tuple)
				    {
				        return tuple.rest;
				    }

				    template <typename Element>
				    LLAMA_FN_HOST_ACC_INLINE auto tupleWithoutFirst(const Tuple<Element>& tuple)
				    {
				        return Tuple<>{};
				    }
				} // namespace llama
				// ==
				// == ./Tuple.hpp ==
				// ============================================================================


			// #include <cstddef>    // amalgamate: file already included
			// #include <string>    // amalgamate: file already included
			// #include <type_traits>    // amalgamate: file already included

			namespace llama::mapping::tree
			{
			    template <typename T>
			    inline constexpr auto one = 1;

			    template <>
			    inline constexpr auto one<boost::mp11::mp_size_t<1>> = boost::mp11::mp_size_t<1>{};

			    template <typename T_Identifier, typename T_Type, typename CountType = std::size_t>
			    struct Leaf
			    {
			        using Identifier = T_Identifier;
			        using Type = T_Type;

			        const CountType count = one<CountType>;
			    };

			    template <typename T_Identifier, typename T_ChildrenTuple, typename CountType = std::size_t>
			    struct Node
			    {
			        using Identifier = T_Identifier;
			        using ChildrenTuple = T_ChildrenTuple;

			        const CountType count = one<CountType>;
			        const ChildrenTuple childs = {};
			    };

			    template <std::size_t ChildIndex = 0, typename ArrayIndexType = std::size_t>
			    struct TreeCoordElement
			    {
			        static constexpr boost::mp11::mp_size_t<ChildIndex> childIndex = {};
			        const ArrayIndexType arrayIndex = {};
			    };

			    template <std::size_t... Coords>
			    using TreeCoord = Tuple<TreeCoordElement<Coords, boost::mp11::mp_size_t<0>>...>;

			    namespace internal
			    {
			        template <typename... Coords, std::size_t... Is>
			        auto treeCoordToString(Tuple<Coords...> treeCoord, std::index_sequence<Is...>) -> std::string
			        {
			            auto s
			                = ((std::to_string(get<Is>(treeCoord).arrayIndex) + ":" + std::to_string(get<Is>(treeCoord).childIndex)
			                    + ", ")
			                   + ...);
			            s.resize(s.length() - 2);
			            return s;
			        }
			    } // namespace internal

			    template <typename TreeCoord>
			    auto treeCoordToString(TreeCoord treeCoord) -> std::string
			    {
			        return std::string("[ ")
			            + internal::treeCoordToString(treeCoord, std::make_index_sequence<tupleSize<TreeCoord>>{})
			            + std::string(" ]");
			    }

			    namespace internal
			    {
			        template <typename Tag, typename RecordDim, typename CountType>
			        struct CreateTreeElement
			        {
			            using type = Leaf<Tag, RecordDim, boost::mp11::mp_size_t<1>>;
			        };

			        template <typename Tag, typename... Fields, typename CountType>
			        struct CreateTreeElement<Tag, Record<Fields...>, CountType>
			        {
			            using type = Node<
			                Tag,
			                Tuple<typename CreateTreeElement<GetFieldTag<Fields>, GetFieldType<Fields>, boost::mp11::mp_size_t<1>>::
			                          type...>,
			                CountType>;
			        };

			        template <typename Leaf, std::size_t Count>
			        struct WrapInNNodes
			        {
			            using type = Node<NoName, Tuple<typename WrapInNNodes<Leaf, Count - 1>::type>>;
			        };

			        template <typename Leaf>
			        struct WrapInNNodes<Leaf, 0>
			        {
			            using type = Leaf;
			        };

			        template <typename RecordDim>
			        using TreeFromRecordDimImpl = typename CreateTreeElement<NoName, RecordDim, std::size_t>::type;
			    } // namespace internal

			    template <typename RecordDim>
			    using TreeFromRecordDim = internal::TreeFromRecordDimImpl<RecordDim>;

			    template <typename ArrayDims, typename RecordDim>
			    using TreeFromDimensions =
			        typename internal::WrapInNNodes<internal::TreeFromRecordDimImpl<RecordDim>, ArrayDims::rank - 1>::type;

			    template <typename RecordDim, typename ArrayDims, std::size_t Pos = 0>
			    LLAMA_FN_HOST_ACC_INLINE auto createTree(const ArrayDims& size)
			    {
			        if constexpr (Pos == ArrayDims::rank - 1)
			            return TreeFromRecordDim<RecordDim>{size[ArrayDims::rank - 1]};
			        else
			        {
			            Tuple inner{createTree<RecordDim, ArrayDims, Pos + 1>(size)};
			            return Node<NoName, decltype(inner)>{size[Pos], inner};
			        }
			    };

			    namespace internal
			    {
			        template <
			            typename ArrayDims,
			            std::size_t... ADIndices,
			            std::size_t FirstRecordCoord,
			            std::size_t... RecordCoords>
			        LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(
			            const ArrayDims& coord,
			            std::index_sequence<ADIndices...>,
			            RecordCoord<FirstRecordCoord, RecordCoords...>)
			        {
			            return Tuple{
			                TreeCoordElement<(ADIndices == ArrayDims::rank - 1 ? FirstRecordCoord : 0)>{coord[ADIndices]}...,
			                TreeCoordElement<RecordCoords, boost::mp11::mp_size_t<0>>{}...,
			                TreeCoordElement<0, boost::mp11::mp_size_t<0>>{}};
			        }
			    } // namespace internal

			    template <typename RecordCoord, typename ArrayDims>
			    LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(const ArrayDims& coord)
			    {
			        return internal::createTreeCoord(coord, std::make_index_sequence<ArrayDims::rank>{}, RecordCoord{});
			    }
			} // namespace llama::mapping::tree
			// ==
			// == ./mapping/tree/TreeFromDimensions.hpp ==
			// ============================================================================


		namespace llama::mapping::tree::functor
		{
		    /// Functor for \ref tree::Mapping. Does nothing with the mapping tree. Is
		    /// used for testing.
		    struct Idem
		    {
		        template <typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const -> Tree
		        {
		            return tree;
		        }

		        template <typename Tree, typename TreeCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree&) const
		            -> TreeCoord
		        {
		            return basicCoord;
		        }

		        template <typename Tree, typename TreeCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree&) const
		            -> TreeCoord
		        {
		            return resultCoord;
		        }
		    };

		    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves,
		    /// creating a SoA layout.
		    struct LeafOnlyRT
		    {
		        template <typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(Tree tree) const
		        {
		            return basicToResultImpl(tree, 1);
		        }

		        template <typename Tree, typename BasicCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            return basicCoordToResultCoordImpl(basicCoord, tree);
		        }

		        template <typename Tree, typename ResultCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree& tree) const
		            -> ResultCoord
		        {
		            return resultCoord;
		        }

		    private:
		        template <typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
		            const Node<Identifier, Type, CountType>& node,
		            std::size_t arraySize)
		        {
		            auto children = tupleTransform(
		                node.childs,
		                [&](auto element) { return basicToResultImpl(element, LLAMA_COPY(node.count) * arraySize); });
		            return Node<Identifier, decltype(children), boost::mp11::mp_size_t<1>>{{}, children};
		        }

		        template <typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
		            const Leaf<Identifier, Type, CountType>& leaf,
		            std::size_t arraySize)
		        {
		            return Leaf<Identifier, Type, std::size_t>{LLAMA_COPY(leaf.count) * arraySize};
		        }

		        template <typename BasicCoord, typename NodeOrLeaf>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
		            const BasicCoord& basicCoord,
		            const NodeOrLeaf& nodeOrLeaf,
		            std::size_t arraySize = 0)
		        {
		            if constexpr (tupleSize<BasicCoord> == 1)
		                return Tuple{TreeCoordElement<BasicCoord::FirstElement::childIndex>{
		                    arraySize + LLAMA_COPY(basicCoord.first.arrayIndex)}};
		            else
		            {
		                const auto& branch = get<BasicCoord::FirstElement::childIndex>(nodeOrLeaf.childs);
		                auto first = TreeCoordElement<BasicCoord::FirstElement::childIndex, boost::mp11::mp_size_t<0>>{};

		                return tupleCat(
		                    Tuple{first},
		                    basicCoordToResultCoordImpl(
		                        basicCoord.rest,
		                        branch,
		                        (arraySize + LLAMA_COPY(basicCoord.first.arrayIndex)) * LLAMA_COPY(branch.count)));
		            }
		        }
		    };

		    namespace internal
		    {
		        template <typename TreeCoord, typename Node>
		        LLAMA_FN_HOST_ACC_INLINE auto getNode(const Node& node)
		        {
		            if constexpr (std::is_same_v<TreeCoord, Tuple<>>)
		                return node;
		            else
		                return getNode<typename TreeCoord::RestTuple>(get<TreeCoord::FirstElement::childIndex>(node.childs));
		        }

		        template <typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
		            const Node<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            if constexpr (std::is_same_v<TreeCoord, Tuple<>>)
		                return Node<Identifier, Type>{newValue, tree.childs};
		            else
		            {
		                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
		                auto replacement = changeNodeRuntime<typename TreeCoord::RestTuple>(current, newValue);
		                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		        }

		        template <typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
		            const Leaf<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            return Leaf<Identifier, Type, std::size_t>{newValue};
		        }

		        struct ChangeNodeChildsRuntimeFunctor
		        {
		            const std::size_t newValue;

		            template <typename Identifier, typename Type, typename CountType>
		            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Node<Identifier, Type, CountType>& element) const
		            {
		                return Node<Identifier, Type, std::size_t>{element.count * newValue, element.childs};
		            }

		            template <typename Identifier, typename Type, typename CountType>
		            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Leaf<Identifier, Type, CountType>& element) const
		            {
		                return Leaf<Identifier, Type, std::size_t>{element.count * newValue};
		            }
		        };

		        template <typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
		            const Node<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            if constexpr (std::is_same_v<TreeCoord, Tuple<>>)
		            {
		                auto children = tupleTransform(tree.childs, ChangeNodeChildsRuntimeFunctor{newValue});
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		            else
		            {
		                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
		                auto replacement = changeNodeChildsRuntime<typename TreeCoord::RestTuple>(current, newValue);
		                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		        }

		        template <typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
		            const Leaf<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            return tree;
		        }
		    } // namespace internal

		    /// Functor for \ref tree::Mapping. Move the run time part of a node one
		    /// level down in direction of the leaves by the given amount (runtime or
		    /// compile time value).
		    /// \tparam TreeCoord tree coordinate in the mapping tree which's run time
		    /// part shall be moved down one level \see tree::Mapping
		    template <typename TreeCoord, typename Amount = std::size_t>
		    struct MoveRTDown
		    {
		        const Amount amount = {};

		        template <typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const
		        {
		            return internal::changeNodeChildsRuntime<TreeCoord>(
		                internal::changeNodeRuntime<TreeCoord>(
		                    tree,
		                    (internal::getNode<TreeCoord>(tree).count + amount - 1) / amount),
		                amount);
		        }

		        template <typename Tree, typename BasicCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            return basicCoordToResultCoordImpl<TreeCoord>(basicCoord, tree);
		        }

		        template <typename Tree, typename ResultCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree&) const
		            -> ResultCoord
		        {
		            return resultCoord;
		        }

		    private:
		        template <typename InternalTreeCoord, typename BasicCoord, typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            if constexpr (std::is_same_v<InternalTreeCoord, Tuple<>>)
		            {
		                if constexpr (std::is_same_v<BasicCoord, Tuple<>>)
		                    return Tuple<>{};
		                else
		                {
		                    const auto& childTree = get<BasicCoord::FirstElement::childIndex>(tree.childs);
		                    const auto rt1 = basicCoord.first.arrayIndex / amount;
		                    const auto rt2
		                        = basicCoord.first.arrayIndex % amount * childTree.count + basicCoord.rest.first.arrayIndex;
		                    auto rt1Child = TreeCoordElement<BasicCoord::FirstElement::childIndex>{rt1};
		                    auto rt2Child = TreeCoordElement<BasicCoord::RestTuple::FirstElement::childIndex>{rt2};
		                    return tupleCat(Tuple{rt1Child}, tupleCat(Tuple{rt2Child}, tupleWithoutFirst(basicCoord.rest)));
		                }
		            }
		            else
		            {
		                if constexpr (InternalTreeCoord::FirstElement::childIndex != BasicCoord::FirstElement::childIndex)
		                    return basicCoord;
		                else
		                {
		                    auto rest = basicCoordToResultCoordImpl<typename InternalTreeCoord::RestTuple>(
		                        tupleWithoutFirst(basicCoord),
		                        get<BasicCoord::FirstElement::childIndex>(tree.childs));
		                    return tupleCat(Tuple{basicCoord.first}, rest);
		                }
		            }
		        }
		    };

		    template <typename TreeCoord, std::size_t Amount>
		    using MoveRTDownFixed = MoveRTDown<TreeCoord, boost::mp11::mp_size_t<Amount>>;
		} // namespace llama::mapping::tree::functor
		// ==
		// == ./mapping/tree/Functors.hpp ==
		// ============================================================================

	// #include "TreeFromDimensions.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/tree/toString.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "TreeFromDimensions.hpp"    // amalgamate: file already expanded

		// #include <boost/core/demangle.hpp>    // amalgamate: file already included
		// #include <string>    // amalgamate: file already included
		#include <typeinfo>

		namespace llama::mapping::tree
		{
		    template <typename T>
		    auto toString(T) -> std::string
		    {
		        return "Unknown";
		    }

		    // handles array indices
		    template <std::size_t I>
		    inline auto toString(RecordCoord<I>) -> std::string
		    {
		        return "";
		    }

		    inline auto toString(NoName) -> std::string
		    {
		        return "";
		    }

		    template <typename... Elements>
		    auto toString(Tuple<Elements...> tree) -> std::string
		    {
		        if constexpr (sizeof...(Elements) > 1)
		            return toString(tree.first) + " , " + toString(tree.rest);
		        else
		            return toString(tree.first);
		    }

		    namespace internal
		    {
		        inline void replace_all(std::string& str, const std::string& search, const std::string& replace)
		        {
		            std::string::size_type i = 0;
		            while ((i = str.find(search, i)) != std::string::npos)
		            {
		                str.replace(i, search.length(), replace);
		                i += replace.length();
		            }
		        }

		        template <typename NodeOrLeaf>
		        auto countAndIdentToString(const NodeOrLeaf& nodeOrLeaf) -> std::string
		        {
		            auto r = std::to_string(nodeOrLeaf.count);
		            if constexpr (std::is_same_v<std::decay_t<decltype(nodeOrLeaf.count)>, std::size_t>)
		                r += "R"; // runtime
		            else
		                r += "C"; // compile time
		            r += std::string{" * "} + toString(typename NodeOrLeaf::Identifier{});
		            return r;
		        }
		    } // namespace internal

		    template <typename Identifier, typename Type, typename CountType>
		    auto toString(const Node<Identifier, Type, CountType>& node) -> std::string
		    {
		        return internal::countAndIdentToString(node) + "[ " + toString(node.childs) + " ]";
		    }

		    template <typename Identifier, typename Type, typename CountType>
		    auto toString(const Leaf<Identifier, Type, CountType>& leaf) -> std::string
		    {
		        auto raw = boost::core::demangle(typeid(Type).name());
		#ifdef _MSC_VER
		        internal::replace_all(raw, " __cdecl(void)", "");
		#endif
		#ifdef __GNUG__
		        internal::replace_all(raw, " ()", "");
		#endif
		        return internal::countAndIdentToString(leaf) + "(" + raw + ")";
		    }
		} // namespace llama::mapping::tree
		// ==
		// == ./mapping/tree/toString.hpp ==
		// ============================================================================


	// #include <type_traits>    // amalgamate: file already included

	namespace llama::mapping::tree
	{
	    namespace internal
	    {
	        template <typename Tree, typename TreeOperationList>
	        struct MergeFunctors
	        {
	        };

	        template <typename Tree, typename... Operations>
	        struct MergeFunctors<Tree, Tuple<Operations...>>
	        {
	            boost::mp11::mp_first<Tuple<Operations...>> operation = {};
	            using ResultTree = decltype(operation.basicToResult(Tree()));
	            ResultTree treeAfterOp;
	            MergeFunctors<ResultTree, boost::mp11::mp_drop_c<Tuple<Operations...>, 1>> next = {};

	            MergeFunctors() = default;

	            LLAMA_FN_HOST_ACC_INLINE
	            MergeFunctors(const Tree& tree, const Tuple<Operations...>& treeOperationList)
	                : operation(treeOperationList.first)
	                , treeAfterOp(operation.basicToResult(tree))
	                , next(treeAfterOp, tupleWithoutFirst(treeOperationList))
	            {
	            }

	            LLAMA_FN_HOST_ACC_INLINE
	            auto basicToResult(const Tree& tree) const
	            {
	                if constexpr (sizeof...(Operations) > 1)
	                    return next.basicToResult(treeAfterOp);
	                else if constexpr (sizeof...(Operations) == 1)
	                    return operation.basicToResult(tree);
	                else
	                    return tree;
	            }

	            template <typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& tree) const
	            {
	                if constexpr (sizeof...(Operations) >= 1)
	                    return next.basicCoordToResultCoord(
	                        operation.basicCoordToResultCoord(basicCoord, tree),
	                        treeAfterOp);
	                else
	                    return basicCoord;
	            }

	            template <typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& tree) const
	            {
	                if constexpr (sizeof...(Operations) >= 1)
	                    return next.resultCoordToBasicCoord(
	                        operation.resultCoordToBasicCoord(resultCoord, tree),
	                        operation.basicToResult(tree));
	                else
	                    return resultCoord;
	            }
	        };

	        template <typename Tree>
	        struct MergeFunctors<Tree, Tuple<>>
	        {
	            MergeFunctors() = default;

	            LLAMA_FN_HOST_ACC_INLINE
	            MergeFunctors(const Tree&, const Tuple<>& treeOperationList)
	            {
	            }

	            LLAMA_FN_HOST_ACC_INLINE
	            auto basicToResult(const Tree& tree) const
	            {
	                return tree;
	            }

	            template <typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(TreeCoord const& basicCoord, Tree const& tree) const
	                -> TreeCoord
	            {
	                return basicCoord;
	            }

	            template <typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(TreeCoord const& resultCoord, Tree const& tree) const
	                -> TreeCoord
	            {
	                return resultCoord;
	            }
	        };

	        template <typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t;

	        template <typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t;

	        template <typename... Children, std::size_t... Is, typename Count>
	        LLAMA_FN_HOST_ACC_INLINE auto getChildrenBlobSize(
	            const Tuple<Children...>& childs,
	            std::index_sequence<Is...> ii,
	            const Count& count) -> std::size_t
	        {
	            return count * (getTreeBlobSize(get<Is>(childs)) + ...);
	        }

	        template <typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t
	        {
	            constexpr std::size_t childCount = boost::mp11::mp_size<std::decay_t<decltype(node.childs)>>::value;
	            return getChildrenBlobSize(node.childs, std::make_index_sequence<childCount>{}, LLAMA_COPY(node.count));
	        }

	        template <typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t
	        {
	            return leaf.count * sizeof(Type);
	        }

	        template <typename Childs, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Childs& childs, const CountType& count) -> std::size_t
	        {
	            return getTreeBlobSize(Node<NoName, Childs, CountType>{count, childs});
	        }

	        template <std::size_t MaxPos, typename Identifier, typename Type, typename CountType, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE auto sumChildrenSmallerThan(
	            const Node<Identifier, Type, CountType>& node,
	            std::index_sequence<Is...>) -> std::size_t
	        {
	            return ((getTreeBlobSize(get<Is>(node.childs)) * (Is < MaxPos)) + ...);
	        }

	        template <typename Tree, typename... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobByte(const Tree& tree, const Tuple<Coords...>& treeCoord)
	            -> std::size_t
	        {
	            const auto firstArrayIndex = treeCoord.first.arrayIndex;
	            if constexpr (sizeof...(Coords) > 1)
	            {
	                constexpr auto firstChildIndex = decltype(treeCoord.first.childIndex)::value;
	                return getTreeBlobSize(tree.childs, firstArrayIndex)
	                    + sumChildrenSmallerThan<firstChildIndex>(
	                           tree,
	                           std::make_index_sequence<tupleSize<typename Tree::ChildrenTuple>>{})
	                    + getTreeBlobByte(get<firstChildIndex>(tree.childs), treeCoord.rest);
	            }
	            else
	                return sizeof(typename Tree::Type) * firstArrayIndex;
	        }
	    } // namespace internal

	    /// An experimental attempt to provide a general purpose description of a
	    /// mapping. \ref ArrayDims and record dimension are represented by a compile
	    /// time tree data structure. This tree is mapped into memory by means of a
	    /// breadth-first tree traversal. By specifying additional tree operations,
	    /// the tree can be modified at compile time before being mapped to memory.
	    template <typename T_ArrayDims, typename T_RecordDim, typename TreeOperationList>
	    struct Mapping
	    {
	        using ArrayDims = T_ArrayDims;
	        using RecordDim = T_RecordDim;
	        using BasicTree = TreeFromDimensions<ArrayDims, RecordDim>;
	        // TODO, support more than one blob
	        static constexpr std::size_t blobCount = 1;

	        using MergedFunctors = internal::MergeFunctors<BasicTree, TreeOperationList>;

	        ArrayDims arrayDimsSize = {};
	        BasicTree basicTree;
	        MergedFunctors mergedFunctors;

	        using ResultTree = decltype(mergedFunctors.basicToResult(basicTree));
	        ResultTree resultTree;

	        Mapping() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        Mapping(ArrayDims size, TreeOperationList treeOperationList, RecordDim = {})
	            : arrayDimsSize(size)
	            , basicTree(createTree<RecordDim>(size))
	            , mergedFunctors(basicTree, treeOperationList)
	            , resultTree(mergedFunctors.basicToResult(basicTree))
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto blobSize(std::size_t const) const -> std::size_t
	        {
	            return internal::getTreeBlobSize(resultTree);
	        }

	        template <std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
	        {
	            auto const basicTreeCoord = createTreeCoord<RecordCoord<RecordCoords...>>(coord);
	            auto const resultTreeCoord = mergedFunctors.basicCoordToResultCoord(basicTreeCoord, basicTree);
	            const auto offset = internal::getTreeBlobByte(resultTree, resultTreeCoord);
	            return {0, offset};
	        }
	    };
	} // namespace llama::mapping::tree
	// ==
	// == ./mapping/tree/Mapping.hpp ==
	// ============================================================================


#ifdef __NVCC__
#    pragma pop
#endif
// ==
// == ./llama.hpp ==
// ============================================================================

// ============================================================================
// == ./Proofs.hpp ==
// ==
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
// #include "ArrayDimsIndexRange.hpp"    // amalgamate: file already expanded
// #include "Core.hpp"    // amalgamate: file already expanded

namespace llama
{
    namespace internal
    {
        template <typename Mapping, std::size_t... Is, typename ArrayDims>
        constexpr auto blobNrAndOffset(const Mapping& m, RecordCoord<Is...>, ArrayDims ad)
        {
            return m.template blobNrAndOffset<Is...>(ad);
        }

        constexpr auto divRoundUp(std::size_t dividend, std::size_t divisor) -> std::size_t
        {
            return (dividend + divisor - 1) / divisor;
        }
    } // namespace internal

// FIXME: this test is actually not correct, because __cpp_constexpr_dynamic_alloc only guarantees constexpr
// std::allocator
#ifdef __cpp_constexpr_dynamic_alloc
    namespace internal
    {
        template <typename T>
        struct DynArray
        {
            constexpr DynArray() = default;

            constexpr DynArray(std::size_t n)
            {
                data = new T[n]{};
            }

            constexpr ~DynArray()
            {
                delete[] data;
            }

            constexpr void resize(std::size_t n)
            {
                delete[] data;
                data = new T[n]{};
            }

            T* data = nullptr;
        };
    } // namespace internal

    /// Proofs by exhaustion of the array and record dimensions, that all values mapped to memory do not overlap.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    template <typename Mapping>
    constexpr auto mapsNonOverlappingly(const Mapping& m) -> bool
    {
        internal::DynArray<internal::DynArray<std::uint64_t>> blobByteMapped(m.blobCount);
        for (auto i = 0; i < m.blobCount; i++)
            blobByteMapped.data[i].resize(internal::divRoundUp(m.blobSize(i), 64));

        auto testAndSet = [&](auto blob, auto offset) constexpr
        {
            const auto bit = std::uint64_t{1} << (offset % 64);
            if (blobByteMapped.data[blob].data[offset / 64] & bit)
                return true;
            blobByteMapped.data[blob].data[offset / 64] |= bit;
            return false;
        };

        bool collision = false;
        forEachLeaf<typename Mapping::RecordDim>([&](auto coord) constexpr
                                                 {
                                                     if (collision)
                                                         return;
                                                     for (auto ad : ArrayDimsIndexRange{m.arrayDims()})
                                                     {
                                                         using Type
                                                             = GetType<typename Mapping::RecordDim, decltype(coord)>;
                                                         const auto [blob, offset]
                                                             = internal::blobNrAndOffset(m, coord, ad);
                                                         for (auto b = 0; b < sizeof(Type); b++)
                                                             if (testAndSet(blob, offset + b))
                                                             {
                                                                 collision = true;
                                                                 break;
                                                             }
                                                     }
                                                 });
        return !collision;
    }
#endif

    /// Proofs by exhaustion of the array and record dimensions, that at least PieceLength elements are always stored
    /// contiguously.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    template <std::size_t PieceLength, typename Mapping>
    constexpr auto mapsPiecewiseContiguous(const Mapping& m) -> bool
    {
        bool collision = false;
        forEachLeaf<typename Mapping::RecordDim>([&](auto coord) constexpr
                                                 {
                                                     std::size_t flatIndex = 0;
                                                     std::size_t lastBlob = std::numeric_limits<std::size_t>::max();
                                                     std::size_t lastOffset = std::numeric_limits<std::size_t>::max();
                                                     for (auto ad : ArrayDimsIndexRange{m.arrayDims()})
                                                     {
                                                         using Type
                                                             = GetType<typename Mapping::RecordDim, decltype(coord)>;
                                                         const auto [blob, offset]
                                                             = internal::blobNrAndOffset(m, coord, ad);
                                                         if (flatIndex % PieceLength != 0
                                                             && (lastBlob != blob
                                                                 || lastOffset + sizeof(Type) != offset))
                                                         {
                                                             collision = true;
                                                             break;
                                                         }
                                                         lastBlob = blob;
                                                         lastOffset = offset;
                                                         flatIndex++;
                                                     }
                                                 });
        return !collision;
    }
} // namespace llama
// ==
// == ./Proofs.hpp ==
// ============================================================================
